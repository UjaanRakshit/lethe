"""Lethe-disaggregated prefill/decode produces the SAME tokens as
vLLM's native prefix cache on the SAME hit/miss schedule, AND the
decode phase genuinely loads KV from Lethe (not a false green via
recompute).

Three runs, separate subprocesses (isolated vLLM engines):

  NATIVE (control): vLLM's OWN prefix cache. enable_prefix_caching=True,
    no connector. Per prompt: a warm-up generate populates the native
    cache with P's prefix, then a decode generate HITS it. This puts
    the prefix on the cache-HIT side of the boundary — the same
    schedule disagg runs on.
  DISAGG: LetheCacheConnector on, enable_prefix_caching=False. Per
    prompt: a PREFILL phase exports P's KV to Lethe, then a DECODE
    phase imports it from Lethe and generates. Prefix is a cache HIT
    served by Lethe.
  VANILLA (informational): single-pass full recompute, no cache. This
    is on the cache-MISS side of the boundary.

Why NATIVE is the gate, not VANILLA. The claim is NOT bit-identical
ACROSS the cache-hit/cache-miss boundary (attention FP reductions are
non-associative on GPU). The claim IS: given the same set of cache
hits vs misses, Lethe+vLLM produces the same tokens as vLLM serving
those same hits via its native prefix cache. So the apples-to-apples
comparison is DISAGG (Lethe-hit) vs NATIVE (vLLM-hit). Comparing
DISAGG vs VANILLA crosses the boundary and is EXPECTED to drift on
some prompts at late token positions (the classic FP-drift signature).
VANILLA is kept only as an informational record of that boundary
effect.

Assertions:
  * token_ids_DISAGG[i] == token_ids_NATIVE[i] for all prompts (the
    gate). A mismatch fails the test.
  * For prompts with >= 2 whole prefix blocks, DISAGG's decode phase
    reports decode_hit_tokens > 0 — proving Lethe SERVED the KV rather
    than vLLM recomputing (false-green guard).

Diagnostics → tests/correctness/disagg_results.json.

Single-engine role-sequenced disaggregation (one engine, two phases).
Physical two-instance disaggregation is out of scope here. See
disagg/orchestrator.py.

Skips when vllm/torch missing, CUDA unavailable, or lethe_server not
built.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

vllm = pytest.importorskip("vllm", reason="vllm not installed; use .venv-vllm")
torch = pytest.importorskip("torch", reason="torch not installed")


REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD = Path(__file__).parent / "_run_disagg.py"
RESULTS_PATH = Path(__file__).parent / "disagg_results.json"


def _find_server() -> Path | None:
    for c in (
        REPO_ROOT / "build" / "cache_server" / "lethe_server",
        REPO_ROOT / "build" / "cache_server" / "lethe_server.exe",
    ):
        if c.exists():
            return c
    return None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listen(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


_SERVER = _find_server()
_HAS_CUDA = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(_SERVER is None,
                       reason="lethe_server not built; run scripts/build.sh"),
    pytest.mark.skipif(not _HAS_CUDA,
                       reason="CUDA unavailable; this test needs a GPU"),
]


def _run_child(mode: str, lethe_address: str | None) -> dict:
    cmd = [sys.executable, str(CHILD), "--mode", mode]
    if lethe_address:
        cmd += ["--lethe-address", lethe_address]
    env = os.environ.copy()
    # Repo root + client on PYTHONPATH so the child can import `disagg`
    # and `lethe_client` regardless of launch dir.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), str(REPO_ROOT / "client"), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                          text=True, timeout=900, env=env)
    if proc.returncode != 0:
        pytest.fail(
            f"disagg child ({mode}) exited {proc.returncode}\n"
            f"stdout tail: {proc.stdout[-2000:]}\n"
            f"stderr tail: {proc.stderr[-3000:]}"
        )
    json_line = None
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    if json_line is None:
        pytest.fail(f"disagg child ({mode}) produced no JSON\n"
                    f"stdout tail: {proc.stdout[-2000:]}")
    return json.loads(json_line)


def test_disagg_token_identical():
    port = _pick_free_port()
    assert _SERVER is not None
    server = subprocess.Popen(
        [str(_SERVER), "lethe_w9", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    addr = f"127.0.0.1:{port}"
    if not _wait_listen(port, timeout=10.0):
        out = server.stdout.read() if server.stdout else ""
        server.kill()
        pytest.fail(f"lethe_server failed to bind {addr}\n{out}")

    diagnostics: dict = {"model": "google/gemma-3-1b-it", "lethe_address": addr}
    try:
        print("=== NATIVE (vLLM-prefix-cache control, the gate) ===", flush=True)
        run_native = _run_child("native", None)
        print("=== DISAGG (Lethe) ===", flush=True)
        run_b = _run_child("disagg", addr)
        print("=== VANILLA (informational: full recompute) ===", flush=True)
        run_vanilla = _run_child("vanilla", None)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    nat = run_native["results"]
    b = run_b["results"]
    van = run_vanilla["results"]
    assert len(nat) == len(b) == len(van), (
        f"prompt count mismatch native={len(nat)} disagg={len(b)} "
        f"vanilla={len(van)}"
    )

    per_prompt = []
    diverged = []          # DISAGG vs NATIVE (the gate)
    diverged_vanilla = []  # DISAGG vs VANILLA (informational)
    for i in range(len(b)):
        tn = nat[i]["output_token_ids"]
        tb = b[i]["output_token_ids"]
        tv = van[i]["output_token_ids"]
        match_native = (tb == tn)
        match_vanilla = (tb == tv)
        bi = b[i]
        per_prompt.append({
            "prompt_index": i,
            "n_tokens_native": len(tn),
            "n_tokens_disagg": len(tb),
            "match_native": match_native,
            "match_vanilla": match_vanilla,
            "first_diff_native": (next((j for j in range(min(len(tn), len(tb)))
                                        if tn[j] != tb[j]), None)
                                  if not match_native else None),
            "n_prefix_blocks": bi.get("n_prefix_blocks", 0),
            "roundtrip_real_layer_hits": bi.get("roundtrip_real_layer_hits"),
            "decode_hit_tokens": bi.get("decode_hit_tokens", 0),
            "decode_contiguous_hits": bi.get("decode_contiguous_hits", 0),
        })
        if not match_native:
            diverged.append(i)
        if not match_vanilla:
            diverged_vanilla.append(i)

    diagnostics["connector_call_counters"] = run_b.get("connector_call_counters", {})
    diagnostics["per_prompt"] = per_prompt
    diagnostics["diverged_vs_native"] = diverged
    diagnostics["diverged_vs_vanilla"] = diverged_vanilla
    diagnostics["disagg_matches_native"] = (len(diverged) == 0)
    diagnostics["note"] = (
        "Gate = DISAGG vs NATIVE (same cache-hit schedule). "
        "diverged_vs_vanilla is informational — vanilla is full "
        "recompute (cache-MISS side of the boundary), where FP "
        "non-associativity legitimately drifts on some prompts."
    )
    RESULTS_PATH.write_text(json.dumps(diagnostics, indent=2))
    print(f"\ndiagnostics → {RESULTS_PATH}")
    print(f"DISAGG vs NATIVE diverged: {diverged}")
    print(f"DISAGG vs VANILLA diverged (informational): {diverged_vanilla}")

    # Gate 1: DISAGG == NATIVE on the same hit/miss schedule.
    if diverged:
        lines = ["DISAGG != NATIVE (gate FAILED). Per-prompt:"]
        for e in per_prompt:
            lines.append(
                f"  prompt {e['prompt_index']}: match_native={e['match_native']} "
                f"first_diff_native={e['first_diff_native']} "
                f"decode_hit_tokens={e['decode_hit_tokens']}")
        lines.append(f"\nFull diagnostics: {RESULTS_PATH}")
        pytest.fail("\n".join(lines))

    # Gate 2: the decode phase actually hit Lethe (false-green guard).
    # Prompts with >= 2 whole prefix blocks must report
    # decode_hit_tokens > 0. (A <=1-block prompt holds its only block
    # back so >=1 token computes; 0 there is legitimate.)
    multi_block = [e for e in per_prompt if e["n_prefix_blocks"] >= 2]
    assert multi_block, (
        "no prompt had >=2 prefix blocks; cannot prove Lethe load path. "
        "Prompt set too short?"
    )
    no_hit = [e["prompt_index"] for e in multi_block
              if e["decode_hit_tokens"] <= 0]
    assert not no_hit, (
        f"FALSE GREEN: prompts {no_hit} have >=2 prefix blocks but "
        f"decode_hit_tokens==0 — vLLM recomputed instead of loading from "
        f"Lethe. The token-identical pass would be meaningless. See "
        f"{RESULTS_PATH}."
    )
