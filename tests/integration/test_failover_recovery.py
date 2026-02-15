"""Failover recovery within the 3.5s budget.

Verifies the design's recovery-time claim:

    dead_after (3000ms detection)  +  ~500ms re-replication  =  3.5s

Test shape:
  1. Spawn 3 lethe_server subprocesses on loopback (50051/2/3) with a
     shared --peers spec.
  2. Insert N=200 distinct blocks via LetheClient (multi-node mode).
     Each block lands at R=2 across the cluster.
  3. SIGKILL node1 (the middle one, just to make the test independent
     of which node happens to be primary for any given block).
  4. Poll the surviving nodes (node0 + node2) every 100ms via direct
     single-node Lookups; record the wall-clock moment when EVERY
     block returns LocalHit from BOTH survivors (R=2 on the surviving
     2-node cluster).
  5. Repeat the cycle 3 times; take the median wall-clock.

Acceptance: median ≤ 3.5s. If borderline (3.5-5s): re-run more, decide
on distribution. If >5s consistently the budget split is under threat
and needs triage — do NOT silently widen dead_after to make the test
pass.

Subprocess lifecycle: spawn the orchestrator processes in a new
session so SIGINT to the test wraps to the cluster. Teardown sends
SIGTERM and SIGKILL-sweeps any straggling lethe_server processes.

Skips when:
  - lethe_server binary isn't built.
  - vllm/lethe_client not importable (the client lib is required to
    drive the test).
"""
from __future__ import annotations

import hashlib
import os
import signal
import socket
import statistics
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO_ROOT / "build" / "cache_server" / "lethe_server"

PORTS = (50061, 50062, 50063)   # different from test_three_node_python.py
NODES = ("fover0", "fover1", "fover2")
HOST = "127.0.0.1"

# Budget per the design: dead_after (3s) + ~500ms re-replication.
BUDGET_SECONDS = 3.5

# Hard cap so a misbehaving run doesn't block CI forever. >5s is the
# threshold above which the budget split needs triage.
MAX_WAIT_SECONDS = 8.0

NUM_BLOCKS = 200
BLOCK_SIZE = 256

requires_built = pytest.mark.skipif(
    not SERVER_BIN.exists(),
    reason=f"lethe_server not built at {SERVER_BIN}",
)


def _wait_for_listen(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def _spawn_node(node_id: str, port: int, peers_spec: str) -> subprocess.Popen:
    # Per-node log file so we can post-mortem re-replication failures.
    log_path = Path(f"/tmp/lethe-fover-{node_id}.log")
    env = os.environ.copy()
    # LETHE_DEBUG_REREP=1 turns on stderr diagnostic prints in
    # replication.cpp's TriggerReReplication + worker loop. Useful
    # for debugging the failover test; harmless in normal runs.
    env["LETHE_DEBUG_REREP"] = "1"
    log_f = open(log_path, "w")
    return subprocess.Popen(
        [str(SERVER_BIN), node_id, str(port), "--peers", peers_spec],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )


def _spawn_cluster() -> list[subprocess.Popen]:
    # Pre-clean: kill any orphan lethe_server processes from a prior run.
    subprocess.run(
        ["pkill", "-KILL", "-f", "lethe_server"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        check=False,
    )
    time.sleep(0.3)

    procs: list[subprocess.Popen] = []
    for i, (node_id, port) in enumerate(zip(NODES, PORTS)):
        peers_spec = ",".join(
            f"{NODES[j]}@{HOST}:{PORTS[j]}" for j in range(3) if j != i
        )
        procs.append(_spawn_node(node_id, port, peers_spec))

    for port in PORTS:
        if not _wait_for_listen(port, timeout=15.0):
            _teardown_cluster(procs)
            pytest.fail(f"node bound to port {port} failed to listen")
    return procs


def _teardown_cluster(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass
    deadline = time.monotonic() + 3.0
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass
    # Sweep stragglers.
    subprocess.run(
        ["pkill", "-KILL", "-f", "lethe_server"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        check=False,
    )


def _make_block(seed: int):
    from lethe_client.client import BlockId
    h = hashlib.sha256(seed.to_bytes(8, "little")).digest()
    payload = hashlib.sha256(seed.to_bytes(8, "little")).digest() * (
        BLOCK_SIZE // 32
    )
    return BlockId(hash=h), payload


def _all_survivors_have_block(client_factory, block_ids) -> int:
    """Return the count of blocks that have LocalHit from BOTH survivors
    (node0 + node2). Used by the polling loop to assert R=2 convergence.
    """
    addrs = {NODES[0]: f"{HOST}:{PORTS[0]}", NODES[2]: f"{HOST}:{PORTS[2]}"}
    # Per-node, RAW lookup (no peers configured) so the response is the
    # local view, not a routed-to-primary view.
    locally_present_per_node: dict[str, set[bytes]] = {n: set() for n in addrs}
    for node_id, addr in addrs.items():
        try:
            with client_factory(addr) as c:
                # Lookup in chunks of 50 to avoid one massive RPC.
                CHUNK = 50
                for i in range(0, len(block_ids), CHUNK):
                    res = c.lookup(block_ids[i : i + CHUNK], request_id="fover-poll")
                    for h in res.hits:
                        if h.source_node == node_id:
                            locally_present_per_node[node_id].add(h.block_id.hash)
        except Exception:
            # Surviving node temporarily unreachable (rare in this
            # loopback setup). Treat as "no progress this tick."
            return 0
    # A block "has both survivors" iff present in both sets.
    both = locally_present_per_node[NODES[0]] & locally_present_per_node[NODES[2]]
    return len(both)


def _run_one_cycle(seed_base: int) -> tuple[float, int, int]:
    """Run one failover cycle. Returns (wall_seconds, blocks_converged,
    blocks_total). wall_seconds is the time from kill to full
    convergence; if convergence never reached within MAX_WAIT_SECONDS,
    returns the partial state.
    """
    from lethe_client.client import LetheClient

    def client_factory(addr, peers=None):
        return LetheClient(primary_address=addr, peers=peers)

    procs = _spawn_cluster()
    try:
        addrs = [(NODES[i], f"{HOST}:{PORTS[i]}") for i in range(3)]
        peers = addrs[:]

        # Seed 200 blocks across the cluster.
        blocks = [_make_block(seed_base + i) for i in range(NUM_BLOCKS)]
        block_ids = [bid for bid, _ in blocks]

        with LetheClient(primary_address=addrs[0][1], peers=peers) as c:
            # Insert in chunks; loopback Inserts are fast.
            CHUNK = 50
            for i in range(0, NUM_BLOCKS, CHUNK):
                accepted = c.insert(
                    blocks[i : i + CHUNK],
                    request_id=f"seed-{i}",
                )
                assert accepted == len(blocks[i : i + CHUNK]), (
                    f"seed insert {i}: accepted={accepted} expected="
                    f"{len(blocks[i : i + CHUNK])}"
                )

        # Let async replication settle so the pre-failover state is
        # really R=2 across the cluster.
        time.sleep(0.5)

        # Hard-kill node1.
        try:
            os.killpg(os.getpgid(procs[1].pid), signal.SIGKILL)
        except Exception as e:
            pytest.fail(f"failed to SIGKILL node1: {e}")
        kill_t = time.monotonic()

        # Poll until R=2 holds on the surviving 2-node cluster or budget
        # blows past MAX_WAIT_SECONDS.
        def factory(addr):
            return LetheClient(primary_address=addr)

        deadline = kill_t + MAX_WAIT_SECONDS
        converged = 0
        while time.monotonic() < deadline:
            converged = _all_survivors_have_block(factory, block_ids)
            if converged >= NUM_BLOCKS:
                wall = time.monotonic() - kill_t
                return wall, converged, NUM_BLOCKS
            time.sleep(0.1)

        # Timed out before convergence.
        wall = time.monotonic() - kill_t
        return wall, converged, NUM_BLOCKS
    finally:
        _teardown_cluster(procs)


@requires_built
def test_failover_recovery_meets_budget():
    """The 3.5s budget acceptance. Median-of-3 cycles."""
    pytest.importorskip("lethe_client.client")

    walls: list[float] = []
    final_convergence: list[tuple[int, int]] = []
    for i in range(3):
        wall, converged, total = _run_one_cycle(seed_base=10_000 * (i + 1))
        walls.append(wall)
        final_convergence.append((converged, total))
        # Brief gap between cycles so the kernel cleans up sockets.
        time.sleep(1.0)

    median_wall = statistics.median(walls)
    print(
        f"\nfailover recovery cycles:\n"
        f"  walls (s): {[round(w, 3) for w in walls]}\n"
        f"  median:    {round(median_wall, 3)}s\n"
        f"  budget:    {BUDGET_SECONDS}s\n"
        f"  convergence per cycle: {final_convergence}\n"
    )

    # All cycles must converge fully — if even one cycle didn't reach
    # NUM_BLOCKS, re-replication is broken, not borderline.
    for i, (converged, total) in enumerate(final_convergence):
        assert converged == total, (
            f"cycle {i}: only {converged}/{total} blocks converged within "
            f"{MAX_WAIT_SECONDS}s — re-replication not draining"
        )

    # The 3.5s budget itself. Median, not max, because environmental
    # noise can spike a single cycle — measure with multiple samples.
    assert median_wall <= BUDGET_SECONDS, (
        f"recovery budget MISSED: median={median_wall:.3f}s vs "
        f"budget={BUDGET_SECONDS}s. This is information to triage, not "
        f"a timer to widen."
    )
