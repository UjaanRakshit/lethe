"""W1.3 acceptance: scheduler-side methods of LetheCacheConnector.

Exercises ``get_num_new_matched_tokens`` (this commit) and
``update_state_after_alloc`` + ``build_connector_meta`` (next commit)
against a real lethe_server subprocess.

Reuses the binary built by W1.2.6 verification at
``build/cache_server/lethe_server``. Skips the whole module when:
  - vllm is not installed (the .venv-vllm gate), or
  - the lethe_server binary hasn't been built.

Threading note: vLLM's scheduler calls the scheduler-side connector
methods from one thread per scheduling iteration (verified at
``vllm/v1/core/sched/scheduler.py:619`` and ``:954`` in 0.19.1 — both
call sites are inside ``Scheduler.schedule()`` and run serially). The
tests construct the connector and call the methods directly without
worrying about concurrency.
"""

from __future__ import annotations

import hashlib
import socket
import subprocess
import time
from pathlib import Path

import pytest

vllm = pytest.importorskip(
    "vllm",
    reason="vllm not installed; install via .venv-vllm — see "
    "docs/decisions/W1_vllm_pin.md",
)

from vllm.config import (  # noqa: E402  — after importorskip
    DeviceConfig,
    KVTransferConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    KVConnectorRole,
)
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.v1.request import Request  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_server_binary() -> Path | None:
    candidates = [
        REPO_ROOT / "build" / "cache_server" / "lethe_server",
        REPO_ROOT / "build" / "cache_server" / "lethe_server.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_listen(host: str, port: int, timeout: float = 5.0) -> bool:
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
    reason="lethe_server binary not found under build/cache_server/. "
    "Build via `bash scripts/build.sh` first.",
)


# ---- Fixtures --------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    """Spawn lethe_server on a free port; tear down at module teardown.

    Same pattern as client/tests/test_client_roundtrip.py — keeps the
    test bootstrap consistent across the two suites.
    """
    port = _pick_free_port()
    assert _SERVER_BIN is not None
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


def _make_connector(address: str, role: KVConnectorRole = KVConnectorRole.SCHEDULER):
    """Build a SCHEDULER-role LetheCacheConnector pointed at `address`.

    Uses the same VllmConfig pattern as
    tests/integration/test_connector_loadability.py.
    """
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    kv = KVTransferConfig(
        kv_connector="LetheCacheConnector",
        kv_connector_module_path="lethe_client.vllm_hook",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lethe_address": address,
            "block_size": 16,
            "model_id": 0,
        },
    )
    cfg = VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=kv,
    )
    return KVConnectorFactory.create_connector(cfg, role)


def _make_request(prompt_token_ids: list[int], request_id: str = "req-1") -> Request:
    """Construct a Request with the minimal fields the scheduler-side
    methods read (request_id, prompt_token_ids). Other fields use
    sane defaults — we never invoke the full scheduling path here.
    """
    return Request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )


def _hashes_for_prefix(token_ids: list[int], block_size: int) -> list[bytes]:
    """Re-derive the chained block hashes in the test, so we can
    pre-insert them server-side and confirm the connector finds them.
    Mirrors LetheCacheConnector.get_num_new_matched_tokens's hash chain.
    """
    from lethe_client.routing import chained_block_hash

    hashes: list[bytes] = []
    running = b"\x00" * 32
    n_blocks = len(token_ids) // block_size
    for i in range(n_blocks):
        start = i * block_size
        running = chained_block_hash(running, token_ids[start : start + block_size])
        hashes.append(running)
    return hashes


def _insert_blocks(client_address: str, hashes: list[bytes]) -> int:
    """Insert one block per hash into the running lethe_server. Payload
    is deterministic so tests can also assert byte-roundtrip if needed.
    Returns the accepted count.
    """
    from lethe_client.client import BlockId, LetheClient

    with LetheClient(primary_address=client_address) as client:
        to_insert = [
            (
                BlockId(hash=h, layer=0, head_group=0, model_id=0),
                hashlib.sha256(h).digest() * 8,  # 256 bytes per block
            )
            for h in hashes
        ]
        return client.insert(to_insert, request_id="seed")


# ---- Tests -----------------------------------------------------------


def test_scheduler_methods_no_lethe_server(caplog):
    """test (a) — graceful degradation when the cache is unreachable.

    Connector points at a port nothing is listening on. ``get_num_new_
    matched_tokens`` must not propagate the gRPC exception; it logs a
    warning and reports zero hits so vLLM continues scheduling.
    """
    dead_address = f"127.0.0.1:{_pick_free_port()}"
    conn = _make_connector(dead_address)

    request = _make_request(prompt_token_ids=list(range(64)))

    import logging

    with caplog.at_level(logging.WARNING, logger="lethe_client.vllm_hook"):
        result = conn.get_num_new_matched_tokens(request, num_computed_tokens=0)

    assert result == (0, False), f"expected (0, False), got {result}"
    assert any(
        "Lethe lookup failed" in rec.message for rec in caplog.records
    ), "expected a warning about the failed lookup"


def test_scheduler_methods_cold_cache(server):
    """test (b) — cold cache, no hits."""
    conn = _make_connector(server)
    request = _make_request(prompt_token_ids=list(range(64)))  # 4 blocks at bs=16

    result = conn.get_num_new_matched_tokens(request, num_computed_tokens=0)
    assert result == (0, False)


def test_scheduler_methods_warm_cache(server):
    """test (c) — first 2 blocks pre-inserted; expect 32 tokens reported."""
    # Use token IDs offset so they don't collide with the cold-cache test's blocks.
    token_ids = list(range(100, 164))  # 64 tokens, 4 blocks
    hashes = _hashes_for_prefix(token_ids, block_size=16)
    assert _insert_blocks(server, hashes[:2]) == 2  # warm the first 2 blocks

    conn = _make_connector(server)
    request = _make_request(prompt_token_ids=token_ids, request_id="warm")

    result = conn.get_num_new_matched_tokens(request, num_computed_tokens=0)
    assert result == (32, False), f"2 hits × 16 tokens expected, got {result}"


def test_scheduler_methods_partial_warm(server):
    """test (d) — hit, miss, hit. Only the first contiguous hit counts."""
    # Distinct token range to keep the cache state of other tests isolated.
    token_ids = list(range(200, 264))  # 64 tokens, 4 blocks
    hashes = _hashes_for_prefix(token_ids, block_size=16)
    # Insert block 0 and block 2 — leave a gap at block 1.
    assert _insert_blocks(server, [hashes[0], hashes[2]]) == 2

    conn = _make_connector(server)
    request = _make_request(prompt_token_ids=token_ids, request_id="partial")

    result = conn.get_num_new_matched_tokens(request, num_computed_tokens=0)
    assert result == (16, False), f"contiguous prefix = block 0 only; got {result}"


def test_get_num_clamps_negative(server):
    """test (f) — contrived: vLLM's local cache already covers more
    tokens than Lethe knows about. The return must be (0, False), NOT
    a negative count, to keep the scheduler's arithmetic sane.
    """
    # Use a token range we know is NOT in the cache (cold for Lethe).
    token_ids = list(range(900, 964))  # 64 tokens, 4 blocks
    conn = _make_connector(server)
    request = _make_request(prompt_token_ids=token_ids, request_id="clamp")

    # Lethe sees 0 hits → hit_tokens = 0. Caller claims 16 tokens
    # already locally computed. 0 - 16 = -16; clamp to 0.
    result = conn.get_num_new_matched_tokens(request, num_computed_tokens=16)
    assert result == (0, False), (
        f"expected clamp to (0, False) when local > external; got {result}"
    )
