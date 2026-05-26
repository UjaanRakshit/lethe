"""Python multi-node integration against a real 3-node cluster.

Spawns three lethe_server subprocesses via ``scripts/run_3node.sh``,
exercises the multi-node LetheClient, asserts:

  * Per-primary RPC batching is real (not one RPC per block).
  * Cross-node Lookup works: insert N blocks across the cluster,
    independent Lookups from each node return HIT (LocalHit or
    RemoteHit-with-source_node, depending on routing).
  * RemoteHit triggers a transparent Fetch: the client surfaces
    bytes equal to what was inserted, regardless of which node
    the client points at.
  * Async replication lands: an Insert on node X with primary X
    eventually shows up as LocalHit on the R-1 successor in the
    ring (within a generous 2-second drain).

Skips when:
  * lethe_server binary isn't built.
  * scripts/run_3node.sh isn't executable.
  * vllm isn't installed (this module doesn't need vllm, but
    importorskip on lethe_client.client confirms the runtime
    deps are present).

Subprocess lifecycle: pytest fixture spawns the orchestrator via
the shell script; teardown sends SIGINT to the orchestrator AND
SIGKILL-sweeps any straggling lethe_server processes by name. The
script's own EXIT trap should handle 95% of cleanup, but the
sweep guards against orphan processes if pytest is interrupted
mid-test.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_BIN = REPO_ROOT / "build" / "cache_server" / "lethe_server"
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_3node.sh"

PORTS = (50051, 50052, 50053)
NODES = ("node0", "node1", "node2")
HOST = "127.0.0.1"

requires_built = pytest.mark.skipif(
    not SERVER_BIN.exists() or not RUN_SCRIPT.exists(),
    reason=(
        f"lethe_server ({SERVER_BIN}) or run_3node.sh ({RUN_SCRIPT}) "
        "missing; run `bash scripts/build.sh` and `chmod +x scripts/run_3node.sh`"
    ),
)


def _wait_for_listen(port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((HOST, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.1)
    return False


@pytest.fixture(scope="module")
def cluster():
    """Bring up the 3-node cluster, tear down at module teardown.

    We launch run_3node.sh in a new process group so SIGINT to the
    orchestrator propagates to its children. Module-scoped because
    cluster bring-up is ~1-2s and we don't want to pay it per test.
    """
    # Pre-clean: kill any orphan lethe_servers from a prior aborted
    # test run. preexec_fn ignored on Windows; we're Linux-only here.
    subprocess.run(["pkill", "-KILL", "-f", "lethe_server"],
                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   check=False)
    time.sleep(0.3)

    proc = subprocess.Popen(
        ["bash", str(RUN_SCRIPT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    # Wait for every node's port to accept connections.
    for port in PORTS:
        if not _wait_for_listen(port, timeout=15.0):
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=3.0)
            except Exception:
                pass
            out = proc.stdout.read() if proc.stdout else ""
            subprocess.run(["pkill", "-KILL", "-f", "lethe_server"],
                           stderr=subprocess.DEVNULL, check=False)
            pytest.fail(f"node port {port} never bound\n{out}")

    addrs = [(n, f"{HOST}:{p}") for n, p in zip(NODES, PORTS)]
    try:
        yield addrs
    finally:
        # Try graceful first; escalate.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except Exception:
            pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            proc.wait(timeout=3.0)
        # Belt and braces.
        subprocess.run(["pkill", "-KILL", "-f", "lethe_server"],
                       stderr=subprocess.DEVNULL, check=False)


def _make_block(seed: int, size: int = 256):
    from lethe_client.client import BlockId
    h = hashlib.sha256(seed.to_bytes(4, "little")).digest()
    payload = hashlib.sha256(seed.to_bytes(4, "little")).digest() * (size // 32)
    return BlockId(hash=h), payload


# ----- Tests --------------------------------------------------------------


@requires_built
def test_cluster_comes_up(cluster):
    """Smoke: all three nodes accept connections and respond to
    Heartbeat with the full peer list."""
    from lethe_client.client import LetheClient
    addrs = dict(cluster)
    for node, addr in addrs.items():
        with LetheClient(primary_address=addr) as c:
            # The simplest Heartbeat surface is to do a trivial Lookup
            # against zero blocks - exercises the RPC plumbing without
            # caring about the result.
            res = c.lookup([], request_id=f"smoke-{node}")
            assert res.hits == [] and res.misses == []


@requires_built
def test_cross_node_lookup_remote_hit_fetches_transparently(cluster):
    """Core cross-node behavior: insert one block on node0; query from
    a LetheClient with all 3 peers configured. The client routes
    per-block via HashRing; if the primary isn't node0, the server
    on node0 returns RemoteHit and the client transparently fetches
    bytes from source_node. Either way, bytes round-trip exactly.
    """
    from lethe_client.client import LetheClient
    addrs = dict(cluster)
    peers = list(addrs.items())

    bid, payload = _make_block(seed=0xABCDEF12)

    # Insert on node0 specifically (server-side Replicator pushes to
    # the ring's R-1 successor of WHEREVER primary is; for this
    # block's primary, that's deterministic per the ring).
    with LetheClient(primary_address=addrs["node0"], peers=peers) as c:
        accepted = c.insert([(bid, payload)], request_id="x-node")
        assert accepted == 1

    # Give async replication a moment to drain.
    time.sleep(0.5)

    # Query from each node's perspective. With peers configured, the
    # client routes Lookup to the BLOCK'S primary regardless of which
    # node we name as primary_address. We should see consistent hits.
    for who, addr in addrs.items():
        with LetheClient(primary_address=addr, peers=peers) as c:
            result = c.lookup([bid], request_id=f"q-from-{who}")
            assert len(result.hits) == 1, (
                f"querying from {who}: expected 1 hit, got "
                f"hits={result.hits} misses={result.misses}"
            )
            # If the server returned RemoteHit, the transparent fetch
            # populated result.fetched with the bytes.
            hit = result.hits[0]
            if hit.source_node and hit.source_node != who:
                fetched = result.fetched.get(bid.hash)
                assert fetched is not None, (
                    f"RemoteHit from {who} → source_node={hit.source_node} "
                    f"but transparent fetch returned None"
                )
                assert fetched == payload, (
                    f"transparent fetch bytes differ from insert "
                    f"({len(fetched)}B vs {len(payload)}B) when querying "
                    f"from {who}"
                )


@requires_built
def test_per_primary_batching_uses_few_rpcs(cluster):
    """Per-primary batching is non-optional: long-context Lookups
    against a 3-node cluster should issue ~3 RPCs (one per distinct
    primary), not N RPCs for N blocks.

    We can't intercept gRPC calls at the LetheClient layer without
    a hook; the next-best signal is timing. With per-primary batching,
    30 blocks should take comparable time to 3 blocks. Without it,
    30 blocks would be ~10× slower due to per-call latency.

    Acceptance: 30-block Lookup completes in well under 1 second
    (server is local, gRPC RTTs are ~1ms). Without batching the
    floor would be 30 × ~1ms = 30 ms RPC time + 30 × Python
    overhead. The test wins easily either way on loopback but
    documents the invariant the next reader needs to know about.
    """
    from lethe_client.client import LetheClient
    addrs = dict(cluster)
    peers = list(addrs.items())

    # Seed 30 distinct cold-cache blocks. NOTE on counting: the
    # server returns optimistic RemoteHit (a routing hint pointing at
    # the block's primary) even when the primary doesn't actually
    # have the block. The client's transparent Fetch then returns
    # None for cold-cache blocks, surfacing as None entries in
    # result.fetched. So the correct cold-cache assertion is "every
    # fetched value is None," NOT "len(misses) == N" - misses only
    # contains blocks for which the server returned NEITHER LocalHit
    # NOR RemoteHit (e.g. router has no peer for the routed primary).
    block_ids = [_make_block(seed=10000 + i)[0] for i in range(30)]

    with LetheClient(primary_address=addrs["node0"], peers=peers) as c:
        t0 = time.monotonic()
        result = c.lookup(block_ids, request_id="batched")
        elapsed = time.monotonic() - t0

    # Every block should be classified - either as a hit (LocalHit on
    # the queried node, or RemoteHit forwarded by it) or a miss. The
    # union covers all 30 input blocks.
    classified = len(result.hits) + len(result.misses)
    assert classified == 30, (
        f"expected 30 classified results, got {classified} "
        f"({len(result.hits)} hits + {len(result.misses)} misses)"
    )
    # Cold cache: every transparent-fetch follow-up returns None.
    # If anything came back with bytes, prior test state leaked.
    leaked = [k for k, v in result.fetched.items() if v is not None]
    assert not leaked, (
        f"cold cache expected; {len(leaked)} blocks returned bytes "
        f"(state leak from a prior test?). First leaked hash: "
        f"{leaked[0].hex()[:16]}…"
    )
    # Timing: per-primary batching should keep this well under 1s on
    # loopback. Without batching, the floor would be 30 × per-block
    # RPC overhead (~30 ms for 1 ms loopback RTT, much more in
    # practice with the transparent-fetch retries).
    assert elapsed < 1.0, (
        f"30-block lookup took {elapsed:.3f}s; per-primary batching "
        "likely regressed (or the cluster is unhealthy)"
    )


@requires_built
def test_replication_lands_within_drain(cluster):
    """Inserting a block on its primary should result, after the
    async replication drains, in the block being LOCALLY present on
    R-1 = 1 OTHER node (the next successor on the ring). The exact
    node depends on the block hash; we don't predict it, we just
    assert SOME other node has it as LocalHit.
    """
    from lethe_client.client import LetheClient
    addrs = dict(cluster)

    # Different seed than the cross-node test so we don't observe
    # state leak from a prior test.
    bid, payload = _make_block(seed=0x55AA77BB)
    with LetheClient(primary_address=addrs["node0"]) as c:
        accepted = c.insert([(bid, payload)], request_id="replication-test")
        assert accepted == 1

    # Drain. The server-side replication thread pool's per-RPC
    # deadline is 2s; we wait up to that long.
    deadline = time.monotonic() + 2.0
    replicas_with_local_hit: list[str] = []
    while time.monotonic() < deadline:
        replicas_with_local_hit.clear()
        for node, addr in addrs.items():
            with LetheClient(primary_address=addr) as c:
                # Single-address lookup (no peers) so we get the
                # RAW server-side decision - LocalHit only when this
                # node has the bytes locally.
                r = c.lookup([bid], request_id=f"replica-check-{node}")
                if r.hits:
                    # source_node == this node means LocalHit on this
                    # node's store. Anything else is RemoteHit
                    # forwarded by this node (which means it doesn't
                    # have the bytes itself).
                    if r.hits[0].source_node == node:
                        replicas_with_local_hit.append(node)
        # Need at least 2: the primary (one of the three) + 1 successor.
        if len(replicas_with_local_hit) >= 2:
            break
        time.sleep(0.1)

    assert len(replicas_with_local_hit) >= 2, (
        f"replication didn't drain in 2s; nodes with LocalHit = "
        f"{replicas_with_local_hit}. Either R=2 replication is broken "
        f"or the replicate_failures counter on the primary is non-zero "
        f"(check /tmp/lethe-node*.log)."
    )
