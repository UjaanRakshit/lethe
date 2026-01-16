"""W1.4 child script: runs one of {vanilla, connector} configurations
against the canned prompt set, prints a JSON line per prompt to
stdout with token IDs + diagnostics, exits.

Spawned as a subprocess by tests/correctness/test_token_identical.py
so each run has an isolated vLLM engine (no cross-run state leakage).

Usage:
    python _run_vllm_for_w14.py --mode vanilla
    python _run_vllm_for_w14.py --mode connector --lethe-address 127.0.0.1:5005

Determinism note: vLLM 0.19.1 has no general "force deterministic" flag
on Ada (sm_89). VLLM_BATCH_INVARIANT exists but requires sm_90+. We
mitigate by:
  - greedy decoding (temperature=0)
  - one prompt per generate() call (constant batch composition)
  - enforce_eager=True (no CUDA-graph capture)
  - dtype=float16 (Gemma-3 default; consistent across runs)
A single warmup generate() runs before the timed prompts to settle
allocator state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# IMPORTANT: env vars must be set BEFORE vllm import.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# VLLM_BATCH_INVARIANT requires sm_90+; not available on 4060 (8.9).
# Documented in the docstring above.

import torch
from vllm import LLM, SamplingParams


PROMPTS: list[str] = [
    # 1. Short factual.
    "The capital of France is",
    # 2. Short narrative.
    "Once upon a time in a distant land,",
    # 3. Short code.
    "def fibonacci(n):\n    if n <= 1:\n",
    # 4. 3-turn dialog (uses raw text rather than chat template to keep
    #    the test independent of tokenizer chat-template versioning).
    (
        "User: What is 12 times 7?\n"
        "Assistant: 12 times 7 is 84.\n"
        "User: And 84 divided by 4?\n"
        "Assistant: 84 divided by 4 is 21.\n"
        "User: Now multiply that by 5.\n"
        "Assistant:"
    ),
    # 5. Medium factual ~150 tokens. Prefix shared with prompt 10.
    (
        "Photosynthesis is the biological process by which green plants, "
        "algae, and some bacteria convert light energy from the sun into "
        "chemical energy stored in glucose. The process occurs primarily "
        "in the chloroplasts of plant cells, which contain the green "
        "pigment chlorophyll. The overall chemical equation for "
        "photosynthesis can be summarized as follows:"
    ),
    # 6. Medium code completion ~150 tokens.
    (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class TransformerBlock(nn.Module):\n"
        "    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):\n"
        "        super().__init__()\n"
        "        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)\n"
        "        self.norm1 = nn.LayerNorm(dim)\n"
        "        self.norm2 = nn.LayerNorm(dim)\n"
        "        self.ff = nn.Sequential(\n"
        "            nn.Linear(dim, ff_dim),\n"
        "            nn.GELU(),\n"
        "            nn.Linear(ff_dim, dim),\n"
        "        )\n"
        "        self.dropout = nn.Dropout(dropout)\n"
        "\n"
        "    def forward(self, x, mask=None):\n"
    ),
    # 7. Long context ~400 tokens, repeated structure (paragraphs share
    #    a template). Tests the prefix cache on highly-redundant input.
    (
        "Item 1: A wooden chair. Material: oak. Weight: 4.2 kilograms. "
        "Dimensions: 80 by 45 by 45 centimeters. Color: natural brown. "
        "Origin: handmade in Vermont, USA. Notes: light wear on the seat.\n"
        "Item 2: A wooden table. Material: oak. Weight: 18.5 kilograms. "
        "Dimensions: 120 by 80 by 75 centimeters. Color: natural brown. "
        "Origin: handmade in Vermont, USA. Notes: small scratch on the surface.\n"
        "Item 3: A wooden bookshelf. Material: oak. Weight: 35 kilograms. "
        "Dimensions: 180 by 80 by 30 centimeters. Color: natural brown. "
        "Origin: handmade in Vermont, USA. Notes: holds approximately 60 books.\n"
        "Item 4: A wooden bed frame. Material: oak. Weight: 42 kilograms. "
        "Dimensions: 200 by 160 by 35 centimeters. Color: natural brown. "
        "Origin: handmade in Vermont, USA. Notes: queen size, slatted base.\n"
        "Item 5:"
    ),
    # 8. Long context ~400 tokens, varied structure.
    (
        "The history of distributed systems is a story of repeated tradeoffs "
        "between consistency, availability, and partition tolerance. In the "
        "1970s, early systems like the ARPANET demonstrated that networks of "
        "computers could exchange messages reliably, but at the cost of "
        "considerable complexity. The 1980s saw the rise of relational "
        "databases that emphasized strong consistency through ACID "
        "transactions. By the 1990s, the web's growth exposed scaling "
        "limits that pushed engineers toward eventual consistency models. "
        "The CAP theorem, formalized by Eric Brewer in 2000 and proven by "
        "Gilbert and Lynch in 2002, made the tradeoffs explicit. NoSQL "
        "databases that emerged in the late 2000s embraced these tradeoffs, "
        "often sacrificing consistency for availability and partition "
        "tolerance. More recent systems like Spanner have shown that with "
        "specialized hardware (atomic clocks) and global infrastructure, "
        "stronger consistency at scale is achievable. The discussion now "
        "centers on:"
    ),
    # 9. Second 3-turn dialog.
    (
        "User: Recommend a book about distributed systems.\n"
        "Assistant: I'd recommend 'Designing Data-Intensive Applications' by Martin Kleppmann.\n"
        "User: What's the main thesis?\n"
        "Assistant: The book argues that modern systems must be reliable, scalable, and maintainable, and explains the data-handling techniques that make these properties possible.\n"
        "User: Who is it written for?\n"
        "Assistant:"
    ),
    # 10. Shares prefix with prompt 5: same first ~140 tokens, then
    #     diverges. The connector should see a long cache hit here on
    #     run C (warm) if prompt 5 was just generated in the same run.
    (
        "Photosynthesis is the biological process by which green plants, "
        "algae, and some bacteria convert light energy from the sun into "
        "chemical energy stored in glucose. The process occurs primarily "
        "in the chloroplasts of plant cells, which contain the green "
        "pigment chlorophyll. The overall chemical equation for "
        "photosynthesis is often written as 6 CO2 + 6 H2O + light energy →"
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "connector"], required=True)
    parser.add_argument("--lethe-address", default=None)
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=1024)
    args = parser.parse_args()

    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.85,
        block_size=args.block_size,
        seed=args.seed,
    )
    if args.mode == "connector":
        if not args.lethe_address:
            print("error: --lethe-address required for connector mode", file=sys.stderr)
            return 2
        from vllm.config import KVTransferConfig
        llm_kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LetheCacheConnector",
            kv_connector_module_path="lethe_client.vllm_hook",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lethe_address": args.lethe_address,
                "block_size": args.block_size,
                "model_id": 0,
            },
        )

    t_load_start = time.time()
    llm = LLM(**llm_kwargs)
    load_seconds = time.time() - t_load_start
    vram_after_load = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # Warmup generate() to settle the allocator. We discard the output.
    _ = llm.generate(["warmup"], sp)

    # One prompt at a time so batch composition is constant across runs.
    t_gen_start = time.time()
    results: list[dict] = []
    for i, prompt in enumerate(PROMPTS):
        out = llm.generate([prompt], sp)
        token_ids = list(out[0].outputs[0].token_ids)
        results.append({
            "prompt_index": i,
            "prompt_chars": len(prompt),
            "output_token_ids": token_ids,
            "num_output_tokens": len(token_ids),
        })
    gen_seconds = time.time() - t_gen_start
    peak_vram = (
        torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    )

    output = {
        "mode": args.mode,
        "lethe_address": args.lethe_address,
        "model": args.model,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "gen_seconds": gen_seconds,
        "vram_after_load_bytes": int(vram_after_load),
        "peak_vram_bytes": int(peak_vram),
        "results": results,
    }
    # Print as one JSON document (the parent reads stdout into json.loads).
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
