"""W1 acceptance gate: three-way control on Gemma-3-1B.

Per CLAUDE.md rule 2, with the Lethe connector enabled, vLLM outputs
must match what vanilla vLLM produces on the *same hit/miss schedule*.
This test runs three configurations against the same canned prompt
set and asserts that all three produce token-for-token identical
output IDs:

  A. Vanilla vLLM, no Lethe connector.
  B. vLLM + connector, fresh lethe_server (cold cache).
  C. vLLM + connector, SAME lethe_server as B (warm cache).

The three runs happen in separate subprocesses so vLLM's process-
global engine state from one run cannot leak into the next. The
lethe_server subprocess is shared between B and C (cold→warm
transition is the test).

Failure-mode interpretation (per the W1.4 prompt):
  - A != B: connector is corrupting on miss (likely save_kv_layer
    overrunning a non-store-eligible path).
  - A == B but B != C: connector is corrupting on hit
    (serialization bug, shape mismatch, or wrong block slot in
    wait_for_layer_load).
  - A != B != C: fundamentally broken; triage from scratch.
  - A != B != C with divergence only on long prompts: almost
    certainly FP non-determinism in the engine, NOT a Lethe bug —
    the wrong fix here would be to change cache code.

Diagnostics dumped to tests/correctness/w1_4_results.json regardless
of pass/fail.

Skips when:
  - vllm or torch not installed (likely the apt-python pytest path)
  - lethe_server binary not built
  - CUDA unavailable
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
CHILD_SCRIPT = Path(__file__).parent / "_run_vllm_for_w14.py"
RESULTS_PATH = Path(__file__).parent / "w1_4_results.json"


def _find_server_binary() -> Path | None:
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


def _wait_for_listen(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


_SERVER_BIN = _find_server_binary()
_HAS_CUDA = torch.cuda.is_available()

pytestmark = [
    pytest.mark.skipif(
        _SERVER_BIN is None,
        reason="lethe_server binary not found; run scripts/build.sh first",
    ),
    pytest.mark.skipif(
        not _HAS_CUDA,
        reason="CUDA unavailable; W1.4 acceptance gate needs a GPU",
    ),
]


def _run_child(mode: str, lethe_address: str | None = None) -> dict:
    """Run the W1.4 child script as a subprocess and parse its JSON stdout."""
    cmd = [sys.executable, str(CHILD_SCRIPT), "--mode", mode]
    if lethe_address:
        cmd += ["--lethe-address", lethe_address]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"child script ({mode}) exited {proc.returncode}\n"
            f"stdout: {proc.stdout[-2000:]}\n"
            f"stderr: {proc.stderr[-2000:]}"
        )
    # The child prints one JSON document at the end; vLLM prints
    # progress/warnings on stderr or earlier on stdout. Take the last
    # JSON-like line.
    json_line = None
    for line in proc.stdout.splitlines()[::-1]:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
            break
    if json_line is None:
        pytest.fail(
            f"child ({mode}) produced no JSON line on stdout\n"
            f"stdout tail: {proc.stdout[-2000:]}"
        )
    try:
        return json.loads(json_line)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"child ({mode}) JSON parse error: {e}\nline: {json_line[:1000]}"
        )


def _start_server(port: int) -> subprocess.Popen:
    assert _SERVER_BIN is not None
    proc = subprocess.Popen(
        [str(_SERVER_BIN), "lethe_w14", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not _wait_for_listen("127.0.0.1", port, timeout=10.0):
        out = proc.stdout.read() if proc.stdout else ""
        proc.kill()
        pytest.fail(f"lethe_server failed to bind 127.0.0.1:{port}\n{out}")
    return proc


def test_token_identical_three_way_control():
    # Spawn lethe_server (shared between B and C).
    port = _pick_free_port()
    server = _start_server(port)
    addr = f"127.0.0.1:{port}"

    diagnostics = {
        "test_started_at": time.time(),
        "model": "google/gemma-3-1b-it",
        "lethe_address": addr,
        "runs": {},
    }

    try:
        # Run A: vanilla vLLM. Connector is NOT instantiated.
        # Done first so the server is freshly empty at run-B start.
        print("=== run A (vanilla vLLM) ===", flush=True)
        run_a = _run_child("vanilla")
        diagnostics["runs"]["A_vanilla"] = {
            k: v for k, v in run_a.items() if k != "results"
        }

        # Run B: connector + cold lethe_server (just-started, no inserts yet).
        print(f"=== run B (connector, cold cache at {addr}) ===", flush=True)
        run_b = _run_child("connector", lethe_address=addr)
        diagnostics["runs"]["B_cold"] = {
            k: v for k, v in run_b.items() if k != "results"
        }

        # Run C: connector + SAME lethe_server (now warm — B's saves are still there).
        print(f"=== run C (connector, warm cache at {addr}) ===", flush=True)
        run_c = _run_child("connector", lethe_address=addr)
        diagnostics["runs"]["C_warm"] = {
            k: v for k, v in run_c.items() if k != "results"
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    # Per-prompt three-way compare. Collect per-prompt diffs so we can
    # see ALL divergences, not just the first.
    n_prompts = len(run_a["results"])
    assert n_prompts == len(run_b["results"]) == len(run_c["results"]), (
        f"prompt count mismatch across runs: "
        f"A={n_prompts}, B={len(run_b['results'])}, C={len(run_c['results'])}"
    )

    per_prompt = []
    diverged = []
    for i in range(n_prompts):
        a = run_a["results"][i]["output_token_ids"]
        b = run_b["results"][i]["output_token_ids"]
        c = run_c["results"][i]["output_token_ids"]
        ab_match = (a == b)
        bc_match = (b == c)
        ac_match = (a == c)
        per_prompt.append({
            "prompt_index": i,
            "n_tokens_a": len(a),
            "n_tokens_b": len(b),
            "n_tokens_c": len(c),
            "A_eq_B": ab_match,
            "B_eq_C": bc_match,
            "A_eq_C": ac_match,
            "first_diff_AB": (
                next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
                if not ab_match else None
            ),
            "first_diff_BC": (
                next((j for j in range(min(len(b), len(c))) if b[j] != c[j]), None)
                if not bc_match else None
            ),
        })
        if not (ab_match and bc_match):
            diverged.append(i)

    diagnostics["per_prompt"] = per_prompt
    diagnostics["diverged_prompt_indices"] = diverged
    diagnostics["all_identical"] = (len(diverged) == 0)
    diagnostics["test_ended_at"] = time.time()
    diagnostics["wall_seconds"] = (
        diagnostics["test_ended_at"] - diagnostics["test_started_at"]
    )

    # Always dump diagnostics, pass or fail.
    RESULTS_PATH.write_text(json.dumps(diagnostics, indent=2))
    print(f"\nW1.4 diagnostics → {RESULTS_PATH}")

    if diverged:
        # Build a per-prompt summary for the failure message.
        lines = ["Three-way control FAILED. Per-prompt:"]
        for entry in per_prompt:
            lines.append(
                f"  prompt {entry['prompt_index']}: "
                f"A_eq_B={entry['A_eq_B']} B_eq_C={entry['B_eq_C']} "
                f"first_diff_AB={entry['first_diff_AB']} "
                f"first_diff_BC={entry['first_diff_BC']}"
            )
        lines.append(f"\nFull diagnostics: {RESULTS_PATH}")
        pytest.fail("\n".join(lines))
