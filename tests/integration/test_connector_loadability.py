"""W1.1 acceptance: LetheCacheConnector loads through vLLM's factory.

This test verifies that the connector class registered as
``lethe_client.vllm_hook:LetheCacheConnector`` is reachable through
vLLM's normal ``KVConnectorFactory.create_connector`` path — not just
that the Python module imports clean. Import-success is trivially
passable by a class that vLLM would reject; the factory roundtrip is
the actual integration surface.

Skips when vllm is not installed (the W0 dev box doesn't have CUDA, so
vllm isn't pip-installable there). CI nodes with vllm 0.19.1 installed
must pass this test.

Reference: factory mechanism at ``factory.py:43-131`` of vllm 0.19.1
(see ``docs/decisions/W1_vllm_pin.md`` for the pin rationale and line
citations).
"""

from __future__ import annotations

import pytest

# Skip the whole module if vllm isn't installed.
vllm = pytest.importorskip(
    "vllm",
    reason="vllm not installed; install with `pip install vllm==0.19.1`",
)

# These imports use the V1 namespace paths verified during W0 against
# the vllm-0.19.1 source tarball. If a future vllm bump changes the
# layout, this test breaks loudly — that's the intended signal.
from vllm.config import (  # noqa: E402  — after importorskip
    DeviceConfig,
    KVTransferConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (  # noqa: E402
    KVConnectorRole,
)


def _make_vllm_config() -> VllmConfig:
    """Minimal VllmConfig pointing the factory at our out-of-tree
    connector. Pattern lifted from
    ``tests/distributed/test_kvlayout.py`` of vllm 0.19.1.
    """
    kv = KVTransferConfig(
        kv_connector="LetheCacheConnector",
        kv_connector_module_path="lethe_client.vllm_hook",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lethe_address": "localhost:50051",
            "block_size": 16,
            "model_id": 0,
        },
    )
    return VllmConfig(
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=kv,
    )


def test_pin_is_honored():
    """W0 sanity: the test environment is pinned to the version we
    actually inspected in the decision doc. If this fails, the pin in
    ``client/pyproject.toml`` and the line-citations in
    ``docs/decisions/W1_vllm_pin.md`` are out of sync with reality.
    """
    assert vllm.__version__ == "0.19.1", (
        f"expected vllm==0.19.1 (see docs/decisions/W1_vllm_pin.md), "
        f"got {vllm.__version__}"
    )


def test_factory_loads_connector_scheduler_role():
    config = _make_vllm_config()
    conn = KVConnectorFactory.create_connector(
        config, KVConnectorRole.SCHEDULER
    )
    assert conn is not None
    # Confirm the factory landed on OUR class, not some compat shim.
    assert type(conn).__name__ == "LetheCacheConnector"
    assert conn.role is KVConnectorRole.SCHEDULER


def test_factory_loads_connector_worker_role():
    config = _make_vllm_config()
    conn = KVConnectorFactory.create_connector(
        config, KVConnectorRole.WORKER
    )
    assert conn is not None
    assert type(conn).__name__ == "LetheCacheConnector"
    assert conn.role is KVConnectorRole.WORKER


def test_abstract_methods_raise_with_subtask_label():
    """W1.1 stubs must name the sub-task that fills them in. Catches the
    'I'll get to that later' drift the user explicitly warned about.
    """
    config = _make_vllm_config()

    # W1.3 scheduler-side methods + W1.4 worker-side methods all landed
    # for real. Their NotImplementedError markers are gone; the behavior
    # is exercised end-to-end by tests/correctness/test_connector_
    # scheduler.py (scheduler) and tests/correctness/test_token_
    # identical.py + test_save_load_byte_identical.py (worker).
    # Asserting that they STAY implemented is more useful than asserting
    # the gone stub messages — if a future commit accidentally reverts
    # one to NotImplementedError, this catches it.
    sched = KVConnectorFactory.create_connector(config, KVConnectorRole.SCHEDULER)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER)

    sched_method_args = {
        "get_num_new_matched_tokens": dict(request=None, num_computed_tokens=0),
        "update_state_after_alloc": dict(
            request=None, blocks=None, num_external_tokens=0
        ),
        "build_connector_meta": dict(scheduler_output=None),
    }
    worker_method_args = {
        "start_load_kv": dict(forward_context=None),
        "wait_for_layer_load": ("layer0",),
        "save_kv_layer": ("layer0", None, None),
        "wait_for_save": (),
    }

    for (conn, method_args, side_label) in (
        (sched, sched_method_args, "W1.3"),
        (worker, worker_method_args, "W1.4"),
    ):
        for method_name, call_args in method_args.items():
            method = getattr(conn, method_name)
            try:
                if isinstance(call_args, dict):
                    method(**call_args)
                else:
                    method(*call_args)
            except NotImplementedError as e:  # pragma: no cover — sentinel
                pytest.fail(
                    f"{method_name} regressed to NotImplementedError({e!r}); "
                    f"{side_label} implementations should remain in place"
                )
            except Exception:
                # Any other exception (AttributeError on None.x,
                # RuntimeError from missing metadata, etc.) is the
                # expected outcome of calling a real method with None.
                pass


def test_extra_config_round_trip():
    """The connector should pick up extra_config values rather than
    falling back silently to defaults — silent fallbacks were the
    failure mode that gave us the SHA-256 fallback bug in W0.
    """
    config = _make_vllm_config()
    conn = KVConnectorFactory.create_connector(config, KVConnectorRole.SCHEDULER)
    assert conn._lethe_address == "localhost:50051"
    assert conn._block_size == 16
    assert conn._model_id == 0


def test_lethe_connector_metadata_is_a_real_subclass():
    """Per W1 review: scheduler→worker metadata must be a real
    subclass, not a dict-of-everything. The fields are pinned by the
    test so a refactor that "loosens" the metadata into ``Any`` breaks
    here.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorMetadata,
    )

    from lethe_client.vllm_hook import (
        LetheBlockSpec,
        LetheConnectorMetadata,
        LetheRequestPayload,
    )

    assert issubclass(LetheConnectorMetadata, KVConnectorMetadata)

    meta = LetheConnectorMetadata()
    # Explicit field types, not opaque containers.
    assert meta.loads == []
    assert meta.stores == []
    assert meta.block_size == 16

    # A round-trip with one payload exercises the dataclass shape.
    spec = LetheBlockSpec(
        chained_hash=b"\x00" * 32, vllm_block_id=7, is_hit=True
    )
    payload = LetheRequestPayload(request_id="req_42", blocks=[spec])
    meta.loads.append(payload)
    assert meta.loads[0].blocks[0].chained_hash == b"\x00" * 32
    assert meta.loads[0].blocks[0].vllm_block_id == 7
    assert meta.loads[0].blocks[0].is_hit is True
