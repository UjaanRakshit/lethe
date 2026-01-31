"""W9 child script: single-engine role-sequenced disaggregation driver.

Spawned as a subprocess by tests/correctness/test_disagg_token_identical.py
so the vLLM engine is isolated per run.

Modes:
  --mode vanilla   : no connector, single generate() per prompt (control).
  --mode disagg    : LetheCacheConnector on; per prompt run a prefill
                     phase (export KV to Lethe) then a decode phase
                     (import KV from Lethe), and report the decode-phase
                     Lethe hit count + a direct round-trip Lookup.

Both modes build the engine with enable_prefix_caching=False so vLLM's
native prefix cache cannot serve the second pass — Lethe is the only
external KV path. greedy / seed=42 / enforce_eager / bfloat16, matching
the W1.4 determinism recipe.

Prints ONE JSON document on stdout at the end (parent reads the last
JSON-looking line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# Per the W9 prompt. On Ada (sm_89) vLLM has no general force-determinism
# kernel path (VLLM_BATCH_INVARIANT needs sm_90+); determinism here comes
# from greedy + seed + enforce_eager + one-prompt-per-generate. Setting
# this is harmless and documents intent.
os.environ.setdefault("VLLM_FORCE_DETERMINISTIC", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from _run_vllm_for_w14 import PROMPTS  # noqa: E402  shared 10-prompt set


def build_engine(mode: str, lethe_address: str, block_size: int,
                 model: str, max_model_len: int, seed: int) -> LLM:
    kwargs = dict(
        model=model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        block_size=block_size,
        seed=seed,
        # The W9 isolation switch: native prefix cache OFF so Lethe is
        # the only external-cache path the decode phase can hit.
        enable_prefix_caching=False,
    )
    if mode == "disagg":
        from vllm.config import KVTransferConfig
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LetheCacheConnector",
            kv_connector_module_path="lethe_client.vllm_hook",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lethe_address": lethe_address,
                "block_size": block_size,
                "model_id": 0,
            },
        )
    return LLM(**kwargs)


def run_vanilla(llm: LLM, max_tokens: int) -> list[dict]:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    results = []
    for i, prompt in enumerate(PROMPTS):
        out = llm.generate([prompt], sp)
        o = out[0]
        results.append({
            "prompt_index": i,
            "output_token_ids": list(o.outputs[0].token_ids),
            "num_output_tokens": len(o.outputs[0].token_ids),
        })
    return results


def run_disagg(llm: LLM, lethe_address: str, block_size: int,
               max_tokens: int) -> list[dict]:
    from disagg.orchestrator import RoleSequencedDisagg
    from lethe_client import vllm_hook
    from lethe_client.client import BlockId, LetheClient
    from lethe_client.routing import chained_block_hash  # noqa: F401

    seq = RoleSequencedDisagg(llm, block_size=block_size)
    probe_client = LetheClient(primary_address=lethe_address)
    results = []

    for i, prompt in enumerate(PROMPTS):
        # --- Phase 1: prefill (export KV to Lethe) ---
        prefill = seq.prefill(prompt)
        hashes = seq.prefix_block_hashes(prefill.prompt_token_ids)

        # --- Round-trip probe (acceptance B): does Lethe contain P's
        # blocks after prefill? Probe BOTH layer=0 (the connector's
        # scheduler presence-probe key) AND a sweep of real layer ids,
        # so we can tell whether the presence-probe scheme actually
        # matches what save_kv_layer stored.
        probe_layer0 = [BlockId(hash=h, layer=0, head_group=0, model_id=0)
                        for h in hashes]
        r0 = probe_client.lookup(probe_layer0, request_id=f"probe0-{i}")
        layer0_hits = len(r0.hits)

        # Sweep candidate layer ids: derive them the way the connector
        # does, from the engine's actual layer names.
        layer_sweep_hits = _probe_real_layers(llm, probe_client, hashes, i)

        # --- Phase 2: decode (import KV from Lethe) ---
        vllm_hook.SCHEDULER_LOOKUP_LOG.clear()
        decode = seq.decode(prompt, max_tokens=max_tokens)
        # The decode request is the only get_num_new_matched_tokens call
        # in this window. Sum its contiguous-hit tokens.
        decode_log = list(vllm_hook.SCHEDULER_LOOKUP_LOG)
        decode_hit_tokens = sum(e.get("hit_tokens", 0) for e in decode_log)
        decode_contig_hits = sum(e.get("contiguous_hits", 0) for e in decode_log)

        results.append({
            "prompt_index": i,
            "output_token_ids": list(decode.token_ids),
            "num_output_tokens": len(decode.token_ids),
            "n_prefix_blocks": len(hashes),
            "roundtrip_layer0_hits": layer0_hits,
            "roundtrip_real_layer_hits": layer_sweep_hits,
            "decode_hit_tokens": decode_hit_tokens,
            "decode_contiguous_hits": decode_contig_hits,
            "decode_lookup_log": decode_log,
            "prefill_wall_s": prefill.wall_seconds,
            "decode_wall_s": decode.wall_seconds,
        })
    return results


def _probe_real_layers(llm, probe_client, hashes, prompt_index):
    """Probe Lethe for the prefix blocks under the REAL per-layer ids
    the connector would have stored, to ground-truth whether anything
    was saved at all. Returns the max hits found for any single layer
    (a full prefix on one layer == the prefix is genuinely stored)."""
    import hashlib
    from lethe_client.client import BlockId

    # Enumerate layer names from the model. vLLM exposes them on the
    # model runner; fall back to a name pattern if we can't reach them.
    layer_names = _enumerate_layer_names(llm)
    if not layer_names:
        return {"note": "could not enumerate layer names", "max_hits": None}

    def layer_id_for(name: str) -> int:
        return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],
                              "little")

    best = 0
    best_layer = None
    for name in layer_names[:4]:  # sample a few layers; enough to confirm
        lid = layer_id_for(name)
        ids = [BlockId(hash=h, layer=lid, head_group=0, model_id=0)
               for h in hashes]
        r = probe_client.lookup(ids, request_id=f"probeL-{prompt_index}")
        if len(r.hits) > best:
            best = len(r.hits)
            best_layer = name
    return {"max_hits": best, "best_layer": best_layer,
            "n_layers_sampled": min(4, len(layer_names))}


def _enumerate_layer_names(llm):
    """Best-effort: dig the attention layer names out of the engine so
    we can reconstruct the connector's per-layer BlockId.layer values."""
    try:
        # vLLM 0.19.1: the forward-context registry holds layer names,
        # but it's only populated during a forward. Easier: walk the
        # model's named_modules for attention layers.
        engine = llm.llm_engine
        # Deep reach; wrapped in try because internals shift.
        model = engine.model_executor.driver_worker.model_runner.model
        names = []
        for n, _ in model.named_modules():
            if n.endswith(".attn") or ".attn" in n:
                names.append(n)
        return names
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "disagg"], required=True)
    ap.add_argument("--lethe-address", default=None)
    ap.add_argument("--model", default="google/gemma-3-1b-it")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "disagg" and not args.lethe_address:
        print("error: --lethe-address required for disagg mode", file=sys.stderr)
        return 2

    t_load = time.time()
    llm = build_engine(args.mode, args.lethe_address or "", args.block_size,
                       args.model, args.max_model_len, args.seed)
    load_s = time.time() - t_load

    # Warmup (settles allocator; discarded).
    _ = llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=1,
                                                 seed=args.seed))

    if args.mode == "vanilla":
        results = run_vanilla(llm, args.max_tokens)
    else:
        results = run_disagg(llm, args.lethe_address, args.block_size,
                             args.max_tokens)

    out = {
        "mode": args.mode,
        "model": args.model,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "load_seconds": load_s,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0,
        "results": results,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
