"""Capacity-crossover sweep: native single-node prefix cache vs. Lethe.

The claim under test: a distributed prefix cache sustains hit rate when the
working set exceeds a single node's KV memory, where vLLM's own single-node
prefix cache collapses.

Two configs, identical workload, identical GPU + gpu_memory_utilization:
  A  vLLM, native prefix caching ON  — the honest baseline (working set sized
     to exceed single-node KV memory).
  B  same + LetheCacheConnector against 3 loopback lethe_server nodes (R=2),
     with Lethe DRAM raised (LETHE_DRAM_BYTES) so overflow lands in Lethe.

Working-set size is swept by W12_NDIST = the number of distinct 256-token
prefixes, each replayed REPEATS times interleaved. Hit rate is measured on
the warm passes via RequestOutput.num_cached_tokens (for config B this counts
Lethe-served tokens). One (config, n_distinct, rep) per process — building two
engines in one process does not release GPU memory between them.

Budget is set by gpu_memory_utilization, deliberately NOT
num_gpu_blocks_override: the override survives consecutive-identical reuse but
silently breaks native prefix reuse under interleaved access (verified), which
would fake a crossover.

Env: W12_CONFIG=A|B, W12_NDIST, W12_REP, W12_MODEL, W12_GPU_UTIL,
     W12_LETHE_DRAM (bytes/node), W12_OUTDIR, LETHE_SERVER_BIN,
     LETHE_EXTRA_LIB (extra LD_LIBRARY_PATH for the server subprocess; on PACE
     the conda env lib + spack gcc-12.3 lib64 — leave empty on a normal box).

Reference run: PACE ICE, one L40S per SLURM job (--exclusive), gemma-3-1b-it,
util=0.12, W12_NDIST in {32,64,128,256,512,1024}, 3 reps each; results
consolidated by plot_crossover.py.
"""
from __future__ import annotations

import json
import os
import random
import socket
import statistics
import subprocess
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("W12_MODEL", "google/gemma-3-1b-it")
CONFIG = os.environ.get("W12_CONFIG", "A").upper()
REP = int(os.environ.get("W12_REP", "0"))
N_DISTINCT = int(os.environ.get("W12_NDIST", "64"))
UTIL = float(os.environ.get("W12_GPU_UTIL", "0.12"))
LETHE_DRAM = int(os.environ.get("W12_LETHE_DRAM", str(6 * (1 << 30))))  # per-node
OUTDIR = os.path.expanduser(os.environ.get("W12_OUTDIR", os.path.join(_REPO, "benchmarks", "plots")))

BLOCK = 16
PREFIX_BLOCKS = 16
PREFIX_TOKENS = PREFIX_BLOCKS * BLOCK  # 256
REPEATS = 3

SERVER_BIN = os.environ.get(
    "LETHE_SERVER_BIN", os.path.join(_REPO, "build", "cache_server", "lethe_server"))
LNODES = ["node0", "node1", "node2"]
LPORTS = [50051, 50052, 50053]
PEERS = ",".join(f"{LNODES[i]}@127.0.0.1:{LPORTS[i]}" for i in range(3))


def log(*a):
    print("[sweep]", *a, flush=True)


def start_lethe():
    senv = os.environ.copy()
    extra = os.environ.get("LETHE_EXTRA_LIB", "")
    if extra:
        senv["LD_LIBRARY_PATH"] = extra + ":" + senv.get("LD_LIBRARY_PATH", "")
    # Size the Lethe tier past the swept working set so the overflow that spills
    # out of the GPU prefix cache lands in Lethe rather than being evicted there
    # too (the laptop defaults are 1 GiB DRAM/node).
    senv["LETHE_DRAM_BYTES"] = str(LETHE_DRAM)
    procs = []
    for i in range(3):
        lf = open(os.path.join(OUTDIR, f"lethe-{LNODES[i]}.log"), "w")
        procs.append(subprocess.Popen(
            [SERVER_BIN, LNODES[i], str(LPORTS[i]), "--peers", PEERS],
            stdout=lf, stderr=subprocess.STDOUT, env=senv))
    for idx, port in enumerate(LPORTS):
        for _ in range(200):
            if procs[idx].poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    break
            except OSError:
                time.sleep(0.1)
    return procs


def build_workload(n_distinct):
    rng = random.Random(20260529)
    prefixes = [[rng.randrange(10, 90000) for _ in range(PREFIX_TOKENS)]
                for _ in range(n_distinct)]
    stream = []  # (rep, d, prompt_token_ids)
    for rep in range(REPEATS):
        for d in range(n_distinct):
            suffix = [90000 + d % 1000, rep, d % 7 + 1]  # unique tail, shared prefix
            stream.append((rep, d, prefixes[d] + suffix))
    return stream


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    res = {"model": MODEL, "config": CONFIG, "n_distinct": N_DISTINCT, "rep": REP,
           "prefix_tokens": PREFIX_TOKENS, "repeats": REPEATS, "util": UTIL,
           "wss_tokens": N_DISTINCT * PREFIX_TOKENS}
    log(f"config={CONFIG} n_distinct={N_DISTINCT} rep={REP} util={UTIL} "
        f"wss_tokens={res['wss_tokens']}")

    common = dict(model=MODEL, dtype="bfloat16", enforce_eager=True,
                  max_model_len=2048, gpu_memory_utilization=UTIL,
                  block_size=BLOCK, seed=42, limit_mm_per_prompt={"image": 0})

    procs = []
    if CONFIG == "B":
        procs = start_lethe()
        time.sleep(3)
        from vllm.config import KVTransferConfig
        common["kv_transfer_config"] = KVTransferConfig(
            kv_connector="LetheCacheConnector",
            kv_connector_module_path="lethe_client.vllm_hook",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lethe_address": "127.0.0.1:50051", "block_size": BLOCK, "model_id": 0},
        )
    try:
        t = time.time()
        llm = LLM(enable_prefix_caching=True, **common)
        res["load_s"] = round(time.time() - t, 1)
        sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
        _ = llm.generate([{"prompt_token_ids": [1, 2, 3, 4]}], sp, use_tqdm=False)

        from lethe_client import vllm_hook
        vllm_hook.CALL_COUNTERS.clear()
        vllm_hook.SCHEDULER_LOOKUP_LOG.clear()

        per = []
        for rep, d, ids in build_workload(N_DISTINCT):
            t0 = time.monotonic()
            out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
            wall = (time.monotonic() - t0) * 1000
            o = out[0]
            per.append({"rep": rep, "wall_ms": wall,
                        "num_cached": getattr(o, "num_cached_tokens", None),
                        "prompt_len": len(o.prompt_token_ids or ids)})

        warm = [p for p in per if p["rep"] >= 1]
        cached = sum((p["num_cached"] or 0) for p in warm)
        promptt = sum(p["prompt_len"] for p in warm)
        res["hit_rate"] = round(cached / max(1, promptt), 4)
        res["warm_requests"] = len(warm)
        res["ttft_p50_ms"] = round(statistics.median(p["wall_ms"] for p in warm), 1)
        warm_walls = sorted(p["wall_ms"] for p in warm)
        res["ttft_p99_ms"] = round(warm_walls[max(0, int(0.99 * len(warm_walls)) - 1)], 1)
        if CONFIG == "B":
            res["meta_load_blocks"] = vllm_hook.CALL_COUNTERS.get("meta_load_blocks", 0)
            res["meta_store_blocks"] = vllm_hook.CALL_COUNTERS.get("meta_store_blocks", 0)
    finally:
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    out_path = os.path.join(OUTDIR, f"W12_sweep_{CONFIG}_n{N_DISTINCT}_{REP}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    log("WROTE", out_path, "hit_rate", res.get("hit_rate"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
