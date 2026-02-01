"""W9 acceptance gate: disaggregated prefill/decode is token-identical
to vanilla single-pass, AND the decode phase genuinely loads KV from
Lethe (not a false green via recompute).

Two runs, separate subprocesses (isolated vLLM engines):

  Run A (control): vanilla single-pass generate per prompt. No Lethe
    connector. enable_prefix_caching=False, greedy, seed=42.
  Run B (disaggregated): LetheCacheConnector on, enable_prefix_caching
    =False. Per prompt: a PREFILL phase exports P's KV to Lethe, then a
    DECODE phase imports it from Lethe and generates. The KV genuinely
    round-trips through Lethe between phases.

Assertions:
  * token_ids_A[i] == token_ids_B[i] for all 10 prompts (the W9
    correctness gate — cache-equivalence on a fixed hit/miss schedule,
    per CLAUDE.md rule 2). A mismatch is stop-condition 1.
  * For prompts with >= 2 whole prefix blocks, run B's decode phase
    reports decode_hit_tokens > 0 — proving Lethe SERVED the KV rather
    than vLLM recomputing (the false-green guard, stop-condition 2).
    (A 1-block prompt holds its only block back so >=1 token computes,
    so decode_hit_tokens can legitimately be 0 there — see the
    connector's hold-back-last-block logic.)

Diagnostics → tests/correctness/w9_results.json.

This is single-engine role-sequenced disaggregation (one engine, two
phases). Physical two-instance disaggregation is W12/PACE. See
disagg/orchestrator.py and docs/weekly/W9.md.

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
CHILD = Path(__file__).parent / "_run_disagg_for_w9.py"
RESULTS_PATH = Path(__file__).parent / "w9_results.json"


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
                       reason="CUDA unavailable; W9 needs a GPU"),
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
        print("=== run A (vanilla control) ===", flush=True)
        run_a = _run_child("vanilla", None)
        print("=== run B (disaggregated via Lethe) ===", flush=True)
        run_b = _run_child("disagg", addr)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    a = run_a["results"]
    b = run_b["results"]
    assert len(a) == len(b), f"prompt count mismatch A={len(a)} B={len(b)}"

    per_prompt = []
    diverged = []
    for i in range(len(a)):
        ta = a[i]["output_token_ids"]
        tb = b[i]["output_token_ids"]
        match = (ta == tb)
        bi = b[i]
        per_prompt.append({
            "prompt_index": i,
            "n_tokens_a": len(ta),
            "n_tokens_b": len(tb),
            "match": match,
            "first_diff": (next((j for j in range(min(len(ta), len(tb)))
                                 if ta[j] != tb[j]), None) if not match else None),
            "n_prefix_blocks": bi.get("n_prefix_blocks", 0),
            "roundtrip_real_layer_hits": bi.get("roundtrip_real_layer_hits"),
            "decode_hit_tokens": bi.get("decode_hit_tokens", 0),
            "decode_contiguous_hits": bi.get("decode_contiguous_hits", 0),
        })
        if not match:
            diverged.append(i)

    diagnostics["connector_call_counters"] = run_b.get("connector_call_counters", {})
    diagnostics["per_prompt"] = per_prompt
    diagnostics["diverged_prompt_indices"] = diverged
    diagnostics["all_identical"] = (len(diverged) == 0)
    diagnostics["run_a_load_seconds"] = run_a.get("load_seconds")
    diagnostics["run_b_load_seconds"] = run_b.get("load_seconds")
    RESULTS_PATH.write_text(json.dumps(diagnostics, indent=2))
    print(f"\nW9 diagnostics → {RESULTS_PATH}")

    # Gate 1: token-identical (stop-condition 1).
    if diverged:
        lines = ["Disagg token-identical FAILED. Per-prompt:"]
        for e in per_prompt:
            lines.append(
                f"  prompt {e['prompt_index']}: match={e['match']} "
                f"first_diff={e['first_diff']} "
                f"decode_hit_tokens={e['decode_hit_tokens']}")
        lines.append(f"\nFull diagnostics: {RESULTS_PATH}")
        pytest.fail("\n".join(lines))

    # Gate 2: the decode phase actually hit Lethe (stop-condition 2 —
    # false-green guard). Prompts with >= 2 whole prefix blocks must
    # report decode_hit_tokens > 0. (A <=1-block prompt holds its only
    # block back so >=1 token computes; 0 there is legitimate.)
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
