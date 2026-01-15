# W1.4 decision: pick `google/gemma-3-1b-it` as the W1 acceptance model

**Date:** 2026-05-26
**Author:** W1.4 session start, before code
**Status:** approved
**Related:** [`W1_vllm_pin.md`](W1_vllm_pin.md), [`W1_3_cpu_vllm.md`](W1_3_cpu_vllm.md)

## TL;DR

W1.4's three-way control test (vanilla / cold-connector / warm-connector)
needs a small instruction-tuned model that fits the 8 GB 4060 with
headroom for the cache server's host-side allocations. Choosing
`google/gemma-3-1b-it`: HF gated access already approved on this
account; ~2 GB on disk; runs comfortably under `gpu_memory_utilization
=0.85`; supported by vLLM 0.19.1's model registry. The decision is
explicitly revisable for W12 on PACE where higher-VRAM options open up.

## What the model needs to do here

W1.4 isn't a benchmark — it's a correctness gate. The model's job is
to expose any divergence between the vanilla and Lethe-equipped runs:

- Long enough context that the connector's cache-hit path actually
  fires (the canned prompt set goes up to ~400 tokens).
- Real attention math (not a toy / random-init model) so FP
  non-determinism would, if it exists, leak into output tokens.
- Small enough that we can run 3 × 10 prompts in a reasonable wall
  clock on a single laptop GPU.
- Instruction-tuned so the canned dialog / template prompts produce
  meaningful output we can eyeball if anything looks off.

The model size is incidental to "correctness gate." Anything ~1-2B
that fits the 4060 with cache-server headroom would work; pick the
one with the cheapest setup path.

## Options considered

| Model | Size | License / Access | vLLM 0.19.1 support | Rejected because |
|---|---|---|---|---|
| **`google/gemma-3-1b-it`** | 1B | Gated, approved on this HF account 2026-04-24 | Yes (gemma-3 model class shipped well before vllm 0.19.1's release) | **Picked** |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Gated, account approval pending | Yes | License application not yet approved on this account; can't unblock today |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Open (Apache-2.0) | Yes | Equally valid choice; lost the tiebreak only because the Gemma HF approval came in first and we want to spend setup time on the test, not on the auth path. Still a fallback if Gemma access surfaces an issue. |
| `google/gemma-4-e2b-it` | ~2B effective | Open | Unclear — gemma-4 architecture (matformer / nested transformer) was post-vllm-0.19.1 release | Deferred to W12 PACE evaluation; integration risk is real and W1.4 is a correctness gate, not a model exploration |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Gated | Yes | Doesn't fit 4060 8 GB with cache-server headroom at FP16; W12 PACE option |
| `google/gemma-3-4b-it` | 4B | Same gate as 3-1B | Yes | Fits but leaves no headroom under `gpu_memory_utilization=0.85` with the cache server running; the 1B variant has the same architecture for correctness purposes |

## Why this picks the right tradeoffs

- **Access already in place.** The Apr 24 Gemma family approval
  covers `gemma-3-1b-it`. No auth latency to fight.
- **vLLM compatibility.** Gemma-3 ships in vllm's stock model
  registry; no need to add a custom model definition.
- **Architecture is non-trivial enough to expose bugs.** Gemma-3
  uses Grouped-Query Attention (num_kv_heads < num_attention_heads),
  RoPE with custom theta, and sliding-window attention on alternate
  layers. If our save/load serialization gets the head-group layout
  wrong, the divergence will show up in the token IDs — exactly what
  a correctness gate is supposed to catch.
- **Small enough for our hardware.** ~2 GB on disk; ~3 GB GPU memory
  for weights at FP16; ample headroom under
  `gpu_memory_utilization=0.85` (which targets ~6.8 GB of the 8 GB
  card) for KV cache + activations + Lethe server's host-side
  allocations + Python overhead.
- **Iteration-friendly.** First-time download ~2 GB / a couple
  minutes; warm-cache reload near-instant.

## What this DOES NOT commit us to

- W12 benchmarks on PACE will rerun against larger Gemma-3 variants
  (4B, 12B, possibly 27B if VRAM allows) or Llama-3.1-8B once the
  Meta application clears, depending on what's set up on PACE at
  benchmark time. The choice here is for W1.4 only.
- We are not declaring Gemma-3 the "Lethe target model" for
  benchmarking. The cache is model-agnostic; W1.4 only needs one
  concrete model to drive the integration test.
- If Gemma-3 surfaces a vLLM bug in 0.19.1 that blocks W1.4, the
  fallback is Qwen2.5-1.5B (no gate, similar size, similar arch
  shape). Decision doc gets an amendment then.

## Conditions that would force a re-pick

- vLLM 0.19.1's model registry does not actually carry gemma-3
  (would surface immediately on `LLM(model="google/gemma-3-1b-it")`
  — registry miss would raise a clear error).
- Gemma-3 generation under our connector segfaults or hits an
  internal vLLM assertion related to GQA + KV transfer (would
  surface in the three-way control test; document and switch to
  Qwen2.5-1.5B).
- Significantly above 6.8 GB peak VRAM under our config (would
  surface as OOM; document, drop to gemma-3-1b at lower
  `gpu_memory_utilization` or fall back to Qwen2.5-1.5B).

## Pin

Test fixture references the literal string `"google/gemma-3-1b-it"`.
No version pin separate from this — HF serves the model from its
revision system; if upstream re-revises in a breaking way, the
local cache pins the SHA-locked snapshot already downloaded.
