"""Pre-flight: save → wire → load is byte-identical.

Runs before the GPU-driven three-way control test. If this fails, the
three-way control will also fail — running this first answers "is it
serialization or is it the engine?" cheaply.

Two scenarios, in increasing scope:
  1. Pure helper roundtrip: `_tensor_to_bytes(_bytes_to_tensor(...))`
     is identity over (shape, dtype). No server, no GPU.
  2. Connector save → real lethe_server Insert RPC → direct
     LetheClient.fetch → bytes-equal-to-source. Uses the WORKER-role
     connector's actual save_kv_layer + wait_for_save path. CPU
     tensor; no GPU required because the connector's save code does
     ``.cpu().contiguous()`` before serializing.

The load-side inject is deliberately NOT exercised here because
``wait_for_layer_load`` needs a real ``forward_context.no_compile_layers``
to find the layer object — that's available only from inside a real
vLLM engine forward, which is the three-way control test's surface.
"fetch returns the exact bytes we saved" is the load-side correctness
signal: ``_bytes_to_tensor`` is unit-tested in scenario 1, so
save+fetch+deserialize is a complete byte-identical chain.
"""

from __future__ import annotations

import hashlib
import socket
import subprocess
import time
from pathlib import Path

import pytest

# Skip the whole module if torch isn't installed (apt python doesn't
# have torch; .venv-vllm does). The pure helper test only needs torch,
# not vllm — so we don't pytest.importorskip("vllm") here.
torch = pytest.importorskip("torch", reason="torch needed for save/load helpers")


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _wait_for_listen(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


# ---- Scenario 1: pure helper roundtrip (no server, no GPU) -----------


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((16, 4, 64), torch.float16),    # typical per-block KV-ish shape
        ((2, 16, 256), torch.float16),   # mirrors save_kv_layer's
                                          # per-block layout (K|V, page_size, kv_dim)
        ((4, 8), torch.float32),
        ((2, 16, 128), torch.bfloat16),  # bf16 path via uint16 view
    ],
)
def test_tensor_bytes_roundtrip_identity(shape, dtype):
    """The helper pair is a bijection over (shape, dtype)."""
    from lethe_client.vllm_hook import _bytes_to_tensor, _tensor_to_bytes

    src = torch.randn(*shape).to(dtype)
    payload = _tensor_to_bytes(src)
    restored = _bytes_to_tensor(payload, shape, dtype, src.device)

    assert restored.shape == src.shape
    assert restored.dtype == src.dtype
    assert torch.equal(restored, src), (
        f"tensor diverged after bytes roundtrip for shape={shape} dtype={dtype}"
    )


def test_tensor_bytes_size_mismatch_raises():
    """A wrong shape claim at load time must raise — silent reshape
    would mask exactly the class of bug we use this test to catch."""
    from lethe_client.vllm_hook import _bytes_to_tensor, _tensor_to_bytes

    src = torch.randn(8, 16).to(torch.float16)
    payload = _tensor_to_bytes(src)
    with pytest.raises(ValueError, match="size mismatch"):
        _bytes_to_tensor(payload, (4, 16), torch.float16, src.device)


# ---- Scenario 2: save via connector → fetch via client ---------------

_SERVER_BIN = _find_server_binary()

requires_server = pytest.mark.skipif(
    _SERVER_BIN is None,
    reason="lethe_server binary not found under build/cache_server/",
)
# Also requires vllm because the connector ctor needs KVConnectorBase_V1.
requires_vllm = pytest.mark.skipif(
    pytest.importorskip("vllm", reason="vllm not installed") is None,
    reason="vllm not installed",
)


@pytest.fixture(scope="module")
def server():
    port = _pick_free_port()
    assert _SERVER_BIN is not None
    proc = subprocess.Popen(
        [str(_SERVER_BIN), "test_node", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not _wait_for_listen("127.0.0.1", port):
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


def _make_worker_connector(lethe_address: str):
    """A WORKER-role LetheCacheConnector pointed at the running server."""
    from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole

    kv = KVTransferConfig(
        kv_connector="LetheCacheConnector",
        kv_connector_module_path="lethe_client.vllm_hook",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lethe_address": lethe_address,
            "block_size": 16,
            "model_id": 0,
        },
    )
    cfg = VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=kv,
    )
    return KVConnectorFactory.create_connector(cfg, KVConnectorRole.WORKER)


@requires_server
@requires_vllm
def test_save_via_connector_then_fetch_is_byte_identical(server):
    """End-to-end: WORKER connector saves a CPU kv_layer block,
    LetheClient fetches the same BlockId, bytes equal source.

    Does NOT exercise wait_for_layer_load — that's the three-way
    control's surface (needs a real forward_context). What this test
    proves: the save path's per-block extract + serialize + Insert
    chain produces bytes that round-trip exactly through the cache
    server when read back via the Fetch RPC.
    """
    from lethe_client.client import BlockId, LetheClient
    from lethe_client.routing import chained_block_hash
    from lethe_client.vllm_hook import (
        LetheBlockSpec,
        LetheConnectorMetadata,
        LetheRequestPayload,
        _layer_id_for,
        _tensor_to_bytes,
    )

    connector = _make_worker_connector(server)
    assert type(connector).__name__ == "LetheCacheConnector"

    # Synthetic kv_layer: (2 K|V, 4 pages, 16 page_size, 64 kv_dim) fp16.
    # CPU is fine — save_kv_layer copies to CPU anyway.
    kv_layer = torch.randn(2, 4, 16, 64).to(torch.float16)

    # We'll save block index 1 (vllm_block_id=1) with a known chained
    # hash. The chained hash for an empty prev_hash + a few token ids.
    block_tokens = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]
    chained_hash = chained_block_hash(b"\x00" * 32, block_tokens)
    layer_name = "model.layers.7.self_attn"
    layer_id = _layer_id_for(layer_name)

    # Build metadata with one store-eligible block.
    metadata = LetheConnectorMetadata(block_size=16)
    metadata.stores = [
        LetheRequestPayload(
            request_id="byte-identical",
            blocks=[LetheBlockSpec(
                chained_hash=chained_hash,
                vllm_block_id=1,
                is_hit=False,
            )],
        )
    ]

    # Bind, save, wait. Drives the actual save path.
    connector.bind_connector_metadata(metadata)
    try:
        connector.save_kv_layer(
            layer_name=layer_name,
            kv_layer=kv_layer,
            attn_metadata=None,  # save_kv_layer doesn't read attn_metadata
        )
        connector.wait_for_save()
    finally:
        connector.clear_connector_metadata()

    # Independent verification: fetch the same BlockId via LetheClient.
    expected_bytes = _tensor_to_bytes(kv_layer[:, 1, :, :])
    with LetheClient(primary_address=server) as client:
        fetched = client.fetch(BlockId(
            hash=chained_hash,
            layer=layer_id,
            head_group=0,
            model_id=0,
        ))
    assert fetched is not None, "block not found in cache after save+wait_for_save"
    if fetched != expected_bytes:
        # Compute the first byte that differs for the failure message.
        n = min(len(fetched), len(expected_bytes))
        first_diff = next(
            (i for i in range(n) if fetched[i] != expected_bytes[i]),
            None,
        )
        raise AssertionError(
            f"byte mismatch: saved {len(expected_bytes)}B, "
            f"fetched {len(fetched)}B; first-diff offset={first_diff}"
        )
