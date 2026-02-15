"""Single-engine role-sequenced disaggregation driver.

Spawned as a subprocess by tests/correctness/test_disagg_token_identical.py
so the vLLM engine is isolated per run.

Modes:
  --mode vanilla   : no connector, single generate() per prompt (control).
  --mode disagg    : LetheCacheConnector on; per prompt run a prefill
                     phase (export KV to Lethe) then a decode phase
                     (import KV from Lethe), and report the decode-phase
                     Lethe hit count + a direct round-trip Lookup.

vanilla/disagg build the engine with enable_prefix_caching=False so vLLM's
native prefix cache cannot serve the second pass — Lethe is the only
external KV path. greedy / seed=42 / enforce_eager / bfloat16 for
determinism.

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
# On Ada (sm_89) vLLM has no general force-determinism kernel path
# (VLLM_BATCH_INVARIANT needs sm_90+); determinism here comes from
# greedy + seed + enforce_eager + one-prompt-per-generate. Setting this
# is harmless and documents intent.
os.environ.setdefault("VLLM_FORCE_DETERMINISTIC", "1")
# Run the engine core (scheduler) + worker IN THIS PROCESS. vLLM v1
# defaults to a multiprocess engine where the connector instances live
# in a child process — their module-global diagnostics (SCHEDULER_LOOKUP_
# LOG / WORKER_STORE_LOG / CALL_COUNTERS) would then be invisible here.
# Single-process keeps them observable and does not change token output.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# Make `disagg` and the shared prompt set importable regardless of how
# this child is launched (repo root for `import disagg`, this dir for
# _run_vllm_for_w14). lethe_client is pip-installed in the venv.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO_ROOT)
from _run_vllm_for_w14 import PROMPTS  # noqa: E402  shared prompt set


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
        # native prefix cache OFF for vanilla/disagg (Lethe is the only
        # external-cache path); ON for the "native" control so it serves
        # the prefix as a cache HIT on the same schedule as disagg.
        enable_prefix_caching=(mode == "native"),
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


def run_native(llm: LLM, max_tokens: int) -> list[dict]:
    """Control: vLLM's own prefix cache serves the prefix on the same
    hit/miss schedule as disagg. Two phases per prompt mirroring disagg:
    a warm-up generate (max_tokens=1) populates the native cache with
    P's prefix, then a decode generate hits it. This is the
    apples-to-apples comparison ("same set of cache hits vs misses"),
    whereas vanilla full-recompute is on the other side of the cache
    boundary.
    """
    warm = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
    decode = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    results = []
    for i, prompt in enumerate(PROMPTS):
        _ = llm.generate([prompt], warm)        # populate native cache
        out = llm.generate([prompt], decode)    # hit native cache, decode
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
        vllm_hook.WORKER_STORE_LOG.clear()
        prefill = seq.prefill(prompt)
        hashes = seq.prefix_block_hashes(prefill.prompt_token_ids)
        store_log = list(vllm_hook.WORKER_STORE_LOG)

        # --- Round-trip probe: does Lethe contain P's blocks after
        # prefill? Probe layer=0 (the connector's scheduler
        # presence-probe key) AND the real layer_ids the save path used
        # (from WORKER_STORE_LOG), so we can tell whether the
        # presence-probe scheme matches what save_kv_layer stored.
        probe_layer0 = [BlockId(hash=h, layer=0, head_group=0, model_id=0)
                        for h in hashes]
        r0 = probe_client.lookup(probe_layer0, request_id=f"probe0-{i}")
        layer0_hits = len(r0.hits)

        # Probe with a real stored layer_id (if any save happened).
        real_layer_hits = {"max_hits": 0, "n_layers_stored": len(store_log)}
        if store_log and hashes:
            lid = store_log[0]["layer_id"]
            ids = [BlockId(hash=h, layer=lid, head_group=0, model_id=0)
                   for h in hashes]
            rr = probe_client.lookup(ids, request_id=f"probeR-{i}")
            real_layer_hits["max_hits"] = len(rr.hits)
            real_layer_hits["layer_id_probed"] = lid
        layer_sweep_hits = real_layer_hits

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vanilla", "native", "disagg"],
                    required=True)
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
        counters = {}
    elif args.mode == "native":
        results = run_native(llm, args.max_tokens)
        counters = {}
    else:
        results = run_disagg(llm, args.lethe_address, args.block_size,
                             args.max_tokens)
        from lethe_client import vllm_hook
        counters = dict(vllm_hook.CALL_COUNTERS)

    out = {
        "mode": args.mode,
        "model": args.model,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "load_seconds": load_s,
        "vram_peak_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else 0,
        "connector_call_counters": counters,
        "results": results,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
