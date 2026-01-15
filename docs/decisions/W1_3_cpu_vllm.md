# W1.3 decision: defer the CPU `LLM.generate()` smoke test to W1.4

**Date:** 2026-05-26
**Author:** W1.3 session, close-out
**Status:** approved
**Related:** [`W1_vllm_pin.md`](W1_vllm_pin.md)

## TL;DR

W1.3 was scoped to scheduler-side connector work and the prompt
specified CPU vLLM as the iteration model ("GPU work begins at W1.4").
Standard `vllm==0.19.1` from PyPI is a CUDA-only wheel and cannot
fall back to CPU at runtime on Linux. Rather than build `vllm-cpu`
from source (costly) or violate the "GPU at W1.4" rule (scope
discipline), W1.3 closes on the six scheduler-side unit tests; the
end-to-end `LLM.generate()` smoke check moves into W1.4 where it's
the natural acceptance gate anyway.

## Source-of-truth read

All citations are against the source installed at
`/home/syclone/lethe/.venv-vllm/lib/python3.10/site-packages/vllm/`
on 2026-05-26 inside the W1.3 verification venv.

## Why the runtime can't fall back to CPU

`vllm/platforms/__init__.py:165-185` (`cpu_platform_plugin`) returns
the `CpuPlatform` only if one of two conditions holds:

```python
def cpu_platform_plugin() -> str | None:
    is_cpu = False
    try:
        is_cpu = vllm_version_matches_substr("cpu")  # vllm-cpu build
        ...
        if not is_cpu:
            import sys
            is_cpu = sys.platform.startswith("darwin")  # macOS
            ...
```

1. **vllm version string contains "cpu"** — produced by a from-source
   build with the CPU target. Our wheel reports `vllm==0.19.1`, no
   `+cpu` suffix.
2. **`sys.platform.startswith("darwin")`** — macOS. We're on Linux
   WSL2.

Neither applies. The platform auto-detection chain falls through:
TPU (no libtpu) → CUDA (NVML present on this WSL2 box) → CUDA wins.
`VLLM_TARGET_DEVICE=cpu` is a build-time hint, not a runtime override.
`current_platform` resolves to `<vllm.platforms.cuda.NvmlCudaPlatform>`
regardless of any environment toggling at runtime.

## Why vllm-cpu wasn't a good fit for W1.3

Building `vllm-cpu` from source means:

- Cloning the vllm repo at the 0.19.1 tag.
- Pulling intel-extension-for-pytorch (IPEX), OpenMP build headers,
  and the CPU-specific kernel toolchain. These aren't in the
  apt-managed build deps we landed in W1.2.6.
- Running `pip install -e . -v` against a CMake-based extension
  build that takes ~30-60 min on the dev box.
- Producing a `vllm==0.19.1+cpu`-style version. That satisfies the
  CPU plugin's substring check, but the resulting build is also
  measurably different from the CUDA path — different attention
  kernels, different KV-cache memory model. So the smoke test
  would prove the connector survives a CPU-mode engine driving it,
  not the GPU-mode engine W1.4 actually uses.

The smoke test's coverage value vs. its setup cost was unfavorable
given the W1.4 token-identical test re-does it on the path we
actually care about.

## Why we did not just use the GPU we have

The WSL2 environment unexpectedly has an RTX 4060 Mobile (8 GB,
CUDA 13.2 driver) reachable via WSL2's CUDA passthrough; `nvidia-smi`
works and `pynvml` sees one device. We could have spun up Qwen2.5-0.5B
on GPU and run the smoke test there.

We chose not to because the W1.3 prompt was explicit:
> "GPU work begins at W1.4. W1.2 and W1.3 are entirely CPU. W1.3's
>  vLLM work runs CPU-vLLM for scheduler-side development; GPU vLLM
>  (RTX 4060 Mobile, Llama-3.2-1B FP16) is W1.4 onward."

Honoring that scope line matters more than getting an extra smoke
test out of W1.3, especially when W1.4's acceptance gate exercises
the same surface against the same model family with stronger
correctness criteria (token-identical, not just "didn't raise on
the scheduler side").

## What W1.3 closes on instead

The six scheduler-side unit tests in
`tests/correctness/test_connector_scheduler.py`:

| test | what it proves |
|---|---|
| `test_scheduler_methods_no_lethe_server` | Graceful degradation: dead cache → log warning, return (0, False), no exception escapes |
| `test_scheduler_methods_cold_cache` | Cold cache → 0 hits reported |
| `test_scheduler_methods_warm_cache` | Pre-seeded 2 blocks → reports 32 tokens (2 × 16 block_size) |
| `test_scheduler_methods_partial_warm` | Hit-miss-hit pattern → only first contiguous block counted (16 tokens) |
| `test_get_num_clamps_negative` | `hit_tokens < num_computed_tokens` → clamp to 0, not negative |
| `test_update_state_then_build_meta_roundtrip` | State recorded by `update_state_after_alloc` is consumed one-shot by `build_connector_meta`; loads + stores populate correctly with chained-hash continuity |

Plus a deliberately-skipped placeholder
`test_llm_generate_smoke_deferred_to_W1_4` that documents the
deferral inline so anyone reading the suite later sees the
decision without having to find this file.

## What W1.4 picks up

The "real `LLM.generate()` runs the scheduler-side methods without
crashing" assertion folds into W1.4's token-identical test, which
already needs:

- vLLM 0.19.1 running on the actually-present 4060 GPU.
- Llama-3.2-1B FP16 (W1.4's spec model — Qwen 0.5B was W1.3's
  iteration speed choice, and is now moot).
- Connector configured the same way.
- All four worker-side methods implemented (`start_load_kv`,
  `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`) — these
  are W1.4's deliverable.

If the scheduler-side methods crashed under a real engine, the
token-identical test would fail before the worker-side stubs even
fired. So the "scheduler-side survives `LLM.generate()`" coverage
falls out of W1.4 acceptance for free.

## Alternative considered, rejected

| | Option | Why rejected |
|---|---|---|
| 1 | Build `vllm-cpu` from source into `.venv-vllm` | 30–60 min build + new build-dep surface (IPEX, OpenMP); proves a code path (CPU kernels) we don't ship; W1.4 re-covers the same surface on GPU |
| 2 | Use the present 4060 GPU under W1.3 | Violates "GPU work begins at W1.4"; smoke is incidental, not load-bearing for W1.3's actual scope |

## What would force a re-decision

- If W1.4's token-identical test passes the worker-side methods but
  fails the scheduler-side somehow, we'd have wished for the
  earlier smoke test and would add it then.
- If the project later needs a CPU-only CI lane (e.g., for a
  GPU-less GitHub Actions runner running connector unit tests),
  building `vllm-cpu` becomes worth its cost.
- If vLLM ever ships a runtime "force CPU platform" env knob, the
  smoke test becomes cheap to add retroactively.

For now: W1.3 closes on unit tests, W1.4 picks up the engine-level
integration.
