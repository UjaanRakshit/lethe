"""Integration test: byte-identical roundtrip through a running
lethe_server.

This test spawns a real lethe_server binary as a subprocess on a
randomly-chosen TCP port, talks to it via the LetheClient over gRPC,
and verifies:

  * N blocks inserted are reported as accepted.
  * Lookup against the same IDs returns N hits, 0 misses.
  * Fetch returns the same bytes that were inserted, byte-for-byte.
  * Distinct BlockIds produce distinct stored content (no aliasing).
  * Independence-from-mutation: Insert → Fetch (capture bytes B1) →
    Insert a different block → Fetch the first id again (B2) → assert
    B1 == B2 == payload. LookupResult::Entry::local_data owns its bytes
    outright and Fetch always returns a fresh copy, so this is a
    trivially-satisfied invariant — kept as a regression guard against
    any future change that re-introduces borrows on the response path.

Skips when the lethe_server binary can't be found — the build can't
be exercised on every dev box (Protobuf / gRPC C++ not installed), but
the test will run on any properly-equipped CI node.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_server_binary() -> Path | None:
    """Locate the built lethe_server binary. Search a few common
    build-output locations to be friendly to both Linux and Windows
    layouts."""
    candidates = [
        REPO_ROOT / "build" / "cache_server" / "lethe_server",
        REPO_ROOT / "build" / "cache_server" / "lethe_server.exe",
        REPO_ROOT / "build" / "cache_server" / "Release" / "lethe_server.exe",
        REPO_ROOT / "build" / "cache_server" / "Debug" / "lethe_server.exe",
        REPO_ROOT / "build" / "cache_server" / "RelWithDebInfo" / "lethe_server.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _pick_free_port() -> int:
    """Bind a socket to port 0 to get an OS-assigned free port, then
    release. Two tests racing for the same port is possible but rare
    enough we don't bother with a retry loop."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_listen(host: str, port: int, timeout: float = 5.0) -> bool:
    """Poll TCP connect until the server accepts or the timeout fires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


_SERVER_BIN = _find_server_binary()
pytestmark = pytest.mark.skipif(
    _SERVER_BIN is None,
    reason=(
        "lethe_server binary not found under build/cache_server/. "
        "Run `bash scripts/build.sh` (Linux/Mac with protobuf+grpc installed) "
        "before this test."
    ),
)


@pytest.fixture(scope="module")
def server():
    """Spawn lethe_server on a free port; tear down at module teardown."""
    port = _pick_free_port()
    assert _SERVER_BIN is not None  # narrowed by pytestmark
    proc = subprocess.Popen(
        [str(_SERVER_BIN), "test_node", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not _wait_for_listen("127.0.0.1", port, timeout=10.0):
            out = proc.stdout.read() if proc.stdout else ""
            proc.kill()
            pytest.fail(f"lethe_server failed to bind 127.0.0.1:{port}\n{out}")
        yield f"127.0.0.1:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _make_block_id(seed: int, layer: int = 0):
    """Deterministic 32-byte hash from a seed; uses SHA-256 here only
    as a uniqueness source — real content hashing goes via
    `lethe_client.routing.chained_block_hash` (BLAKE3)."""
    from lethe_client.client import BlockId
    return BlockId(
        hash=hashlib.sha256(seed.to_bytes(4, "little")).digest(),
        layer=layer,
    )


def _make_payload(seed: int, size: int) -> bytes:
    """Deterministic byte pattern; ensures distinct seeds → distinct bytes."""
    h = hashlib.sha256(seed.to_bytes(4, "little")).digest()
    # Repeat the 32-byte digest to fill the target size; truncate the tail.
    repeat = (size + 31) // 32
    return (h * repeat)[:size]


def test_insert_lookup_roundtrip(server):
    from lethe_client.client import LetheClient

    N = 8
    payloads: dict[int, bytes] = {}
    block_ids = []
    with LetheClient(primary_address=server) as client:
        # Insert N blocks with distinct ids + distinct payloads.
        to_insert = []
        for i in range(1, N + 1):
            bid = _make_block_id(i)
            payload = _make_payload(i, 512)
            payloads[i] = payload
            block_ids.append(bid)
            to_insert.append((bid, payload))
        accepted = client.insert(to_insert, request_id="r1")
        assert accepted == N, f"expected {N} accepted, got {accepted}"

        # Lookup returns N hits.
        result = client.lookup(block_ids, request_id="r2")
        assert len(result.hits) == N
        assert len(result.misses) == 0
        assert result.hit_rate == 1.0

        # Fetch each block and assert byte-identical roundtrip.
        for i in range(1, N + 1):
            bid = _make_block_id(i)
            got = client.fetch(bid)
            assert got is not None, f"block {i} missing on fetch"
            assert got == payloads[i], (
                f"block {i} bytes diverged: "
                f"sent {len(payloads[i])}B, got {len(got)}B"
            )


def test_distinct_ids_distinct_content(server):
    """Aliasing check: BlockIds that differ ONLY in layer / head_group /
    model_id are stored as distinct entries with distinct content. Those
    fields disambiguate the in-memory map even though they don't enter
    the routing hash."""
    from lethe_client.client import LetheClient

    with LetheClient(primary_address=server) as client:
        # Same hash bytes, different layer/head_group.
        from lethe_client.client import BlockId
        common_hash = hashlib.sha256(b"alias-test").digest()
        b0 = BlockId(hash=common_hash, layer=0, head_group=0)
        b1 = BlockId(hash=common_hash, layer=1, head_group=0)
        b2 = BlockId(hash=common_hash, layer=0, head_group=1)
        p0 = _make_payload(101, 128)
        p1 = _make_payload(102, 128)
        p2 = _make_payload(103, 128)

        accepted = client.insert([(b0, p0), (b1, p1), (b2, p2)], request_id="r3")
        assert accepted == 3

        assert client.fetch(b0) == p0
        assert client.fetch(b1) == p1
        assert client.fetch(b2) == p2


def test_lookup_miss(server):
    from lethe_client.client import LetheClient

    with LetheClient(primary_address=server) as client:
        unknown = _make_block_id(99999)
        result = client.lookup([unknown], request_id="r-miss")
        assert len(result.hits) == 0
        assert len(result.misses) == 1
        assert client.fetch(unknown) is None


def test_local_data_lifetime_contract_via_grpc(server):
    """Server-side Fetch returns immutable byte copies, not a span the
    client could dereference past a subsequent mutation.
    LookupResult::Entry::local_data is an owned vector and Fetch always
    copies the bytes onto the wire, so the test is trivially satisfied.
    Kept as a regression guard against a future change that re-introduces
    borrows on the response path.

    Test scenario:
      1. Insert block X with payload P.
      2. Fetch X → capture bytes B1.
      3. Insert a DIFFERENT block Y (no aliasing with X). This is the
         most-mutating operation the current API exposes; with Erase /
         eviction the test would be sharper.
      4. Fetch X again → bytes B2.
      5. Assert B1 == B2 == P.
    """
    from lethe_client.client import BlockId, LetheClient

    with LetheClient(primary_address=server) as client:
        x = _make_block_id(7000)
        x_payload = _make_payload(7000, 1024)
        client.insert([(x, x_payload)], request_id="r-life-1")

        b1 = client.fetch(x)
        assert b1 == x_payload

        # Now drive a mutation (insert a different block). If the server
        # ever held a stale span across this, the next Fetch could
        # observe garbage; we assert it doesn't.
        y = _make_block_id(7001)
        y_payload = _make_payload(7001, 2048)
        client.insert([(y, y_payload)], request_id="r-life-2")

        b2 = client.fetch(x)
        assert b2 == x_payload, "fetched bytes diverged after intervening mutation"
        assert b2 == b1, "successive fetches of the same block must agree"

        # Capture-then-mutation pattern on the CLIENT side: the bytes
        # we already received in b1 are independent of server state.
        # This is the wire-level enforcement of the contract.
        assert isinstance(b1, bytes)  # bytes is immutable in Python


def test_repeated_insert_is_idempotent(server):
    """The store treats Put on an existing id as a no-op (content-
    addressed; same hash → same bytes). Used-bytes accounting must
    not double on repeat inserts."""
    from lethe_client.client import LetheClient

    with LetheClient(primary_address=server) as client:
        bid = _make_block_id(8000)
        payload = _make_payload(8000, 256)

        accepted_1 = client.insert([(bid, payload)], request_id="idem-1")
        accepted_2 = client.insert([(bid, payload)], request_id="idem-2")

        # First insert is newly accepted (returns 1); the second is a
        # no-op on an existing id (returns 0). LetheCache::Insert
        # counts "newly accepted" blocks; the proto's accepted_count
        # carries that across the wire.
        assert accepted_1 == 1
        assert accepted_2 == 0

        # Roundtrip still works.
        assert client.fetch(bid) == payload
