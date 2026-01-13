"""vLLM ↔ Lethe integration — LetheCacheConnector.

Targets vllm 0.19.1 exactly (see ``docs/decisions/W1_vllm_pin.md``).
Subclasses ``vllm.distributed.kv_transfer.kv_connector.v1.base.KVConnectorBase_V1``
(``base.py:170``); the factory at ``kv_connector/factory.py:43`` instantiates
us twice per engine — once with ``KVConnectorRole.SCHEDULER`` and once with
``KVConnectorRole.WORKER``.

Loading from a vLLM CLI / SDK call (out-of-tree path,
``factory.py:112-123``)::

    --kv-transfer-config '{
        "kv_connector": "LetheCacheConnector",
        "kv_connector_module_path": "lethe_client.vllm_hook",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lethe_address": "lethe-node0:50051",
            "block_size": 16,
            "model_id": 0
        }
    }'

The seven abstract methods are stubbed in this file; each
``NotImplementedError`` names the W1 sub-task that will implement it,
so the dependency graph between sub-tasks lives in the code itself
rather than in a planning document that can rot.

Correctness invariant (CLAUDE.md rule 2, restated): with this connector
enabled, model outputs must be token-for-token identical to vanilla vLLM
on the *same hit/miss schedule* — not bit-identical across the
cache-hit/cache-miss boundary, which is unwinnable on GPU due to
non-associative FP reductions in attention. The W1.4 token-identical
test asserts the equivalence-on-fixed-schedule version.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from .client import BlockId, LetheClient
from .routing import chained_block_hash

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduler → worker metadata payload
# ---------------------------------------------------------------------------


@dataclass
class LetheBlockSpec:
    """One KV block's identifying info on the scheduler→worker wire.

    Mirrors the C++ ``lethe::BlockId`` plus where the block lives in the
    vLLM-allocated paged-KV block table for the request. Explicit fields
    on purpose — the scheduler↔worker boundary is wire-format-adjacent
    and fuzzy schemas there cost more to debug than they save.
    """

    # 32-byte BLAKE3 chained block hash. Matches the C++ Hash256 and the
    # output of ``lethe_client.routing.chained_block_hash``.
    chained_hash: bytes
    # Index into the worker's paged-KV block table for this request.
    # The worker uses this to know which slots to fill on load, or to
    # know which slots to drain on save.
    vllm_block_id: int
    # True iff Lethe reported this hash as cached at scheduler-time.
    # Worker uses this to decide load vs. let-vLLM-recompute.
    is_hit: bool


@dataclass
class LetheRequestPayload:
    """Per-request bundle of block specs. One per scheduled request."""

    request_id: str
    blocks: list[LetheBlockSpec] = field(default_factory=list)


@dataclass
class LetheConnectorMetadata(KVConnectorMetadata):
    """Scheduler → worker payload for one engine step.

    Produced by ``LetheCacheConnector.build_connector_meta`` (scheduler
    role) and consumed by ``start_load_kv`` / ``save_kv_layer`` (worker
    role). All fields are explicitly typed; this is the W1 contract
    between roles and should not grow opaque ``dict[str, Any]``
    payloads without a corresponding decision-doc entry.
    """

    # Requests whose KV blocks should be loaded FROM Lethe into the
    # worker's paged KV buffer before the upcoming forward pass.
    loads: list[LetheRequestPayload] = field(default_factory=list)
    # Requests whose newly-computed KV should be saved TO Lethe AFTER
    # the upcoming forward pass.
    stores: list[LetheRequestPayload] = field(default_factory=list)
    # vLLM block size (tokens per block). Cached here so the worker
    # doesn't have to reach back into VllmConfig.
    block_size: int = 16


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class LetheCacheConnector(KVConnectorBase_V1):
    """vLLM V1 KV transfer connector backed by a Lethe cluster.

    Constructor signature MUST match ``KVConnectorBase_V1.__init__``
    exactly (base.py:183-207) — the factory at ``factory.py:77-82``
    invokes us positionally. Connector-specific configuration is read
    from ``vllm_config.kv_transfer_config.kv_connector_extra_config``
    (a plain ``dict``); no kwargs smuggling.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        extra = (
            self._kv_transfer_config.kv_connector_extra_config or {}
        )  # ``factory.py:106`` → ``KVTransferConfig.kv_connector_extra_config``
        self._lethe_address: str = str(
            extra.get("lethe_address", "localhost:50051")
        )
        # block_size: prefer explicit extra_config override (debug /
        # benchmark), otherwise mirror vLLM's cache_config so the
        # hash-chain block boundaries match exactly.
        if extra.get("block_size") is not None:
            self._block_size = int(extra["block_size"])
        else:
            self._block_size = int(vllm_config.cache_config.block_size)
        self._model_id: int = int(extra.get("model_id", 0))

        # Lazy gRPC channel. The scheduler-role instance does Lookups
        # only; the worker-role instance does Inserts and Streams.
        # Opening the channel on first use keeps engine startup cheap.
        self._client: LetheClient | None = None

        # Scheduler-side bookkeeping. Cleared each step by
        # ``build_connector_meta``. ``Request`` is forward-declared, so
        # we type the value as ``Any`` at runtime.
        self._requests_need_load: dict[str, Any] = {}

        logger.info(
            "LetheCacheConnector init role=%s lethe=%s block_size=%d model_id=%d",
            role.name,
            self._lethe_address,
            self._block_size,
            self._model_id,
        )

    def _ensure_client(self) -> LetheClient:
        if self._client is None:
            self._client = LetheClient(primary_address=self._lethe_address)
        return self._client

    # ======================================================================
    # Worker-side abstract methods (base.py:298-361)
    # ======================================================================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        # W1: synchronous load path; ``False`` is returned from
        # ``get_num_new_matched_tokens`` to signal sync semantics, so this
        # method blocks. W6 turns this into a real async start.
        raise NotImplementedError("W1.4")

    def wait_for_layer_load(self, layer_name: str) -> None:
        raise NotImplementedError("W1.4")

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("W1.4")

    def wait_for_save(self) -> None:
        raise NotImplementedError("W1.4")

    # ======================================================================
    # Scheduler-side abstract methods (base.py:449-518)
    # ======================================================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Count external KV cache tokens available beyond what vLLM
        already has locally.

        Returns ``(new_tokens, False)`` — the ``False`` declares that
        Lethe loads are synchronous (worker-side ``start_load_kv``
        blocks). W6 flips this to ``True`` once RDMA-backed transfer
        produces real futures.

        Threading: called by the scheduler from a single thread per
        scheduling iteration (vllm/v1/core/sched/scheduler.py:619),
        same thread as ``build_connector_meta``. No lock required.
        """
        token_ids = request.prompt_token_ids or []
        if not token_ids:
            return (0, False)

        # Lethe addresses by whole blocks. Partial trailing tokens
        # become a separate request concern — they get recomputed.
        n_blocks = len(token_ids) // self._block_size
        if n_blocks == 0:
            return (0, False)

        # Compute the chained block hashes for the prefix. This mirrors
        # the C++ server's hash chain bit-for-bit
        # (lethe_client/routing.py::chained_block_hash, BLAKE3).
        block_hashes: list[bytes] = []
        running = b"\x00" * 32
        for i in range(n_blocks):
            start = i * self._block_size
            block_tokens = token_ids[start : start + self._block_size]
            running = chained_block_hash(running, block_tokens)
            block_hashes.append(running)

        # Batch into one Lookup RPC. All blocks for one prefix route to
        # the same primary per the W0 routing policy (BlockId.layer is
        # not in the hash) so the batched RPC is the natural shape.
        # W1 single-node: layer=0 is fine as a presence probe.
        probe_ids = [
            BlockId(
                hash=h,
                layer=0,
                head_group=0,
                model_id=self._model_id,
            )
            for h in block_hashes
        ]

        # Graceful degradation: if the cache is unreachable, fall back
        # to "no hits" — never block scheduling on cache liveness.
        try:
            client = self._ensure_client()
            result = client.lookup(
                probe_ids,
                request_id=request.request_id,
                requesting_node="lethe-scheduler",
            )
        except Exception as e:  # noqa: BLE001 — grpc raises a wide tree
            logger.warning(
                "Lethe lookup failed; falling back to cold-cache scheduling. "
                "request_id=%s lethe_address=%s err_type=%s err=%s",
                request.request_id,
                self._lethe_address,
                type(e).__name__,
                e,
            )
            return (0, False)

        # Count contiguous hits from the START of the prefix.
        # A hit at block index 5 with a miss at index 3 doesn't help —
        # vLLM needs contiguous prefix to load any of it.
        hit_set: set[bytes] = {bytes(h.block_id.hash) for h in result.hits}
        contiguous_hits = 0
        for h in block_hashes:
            if h in hit_set:
                contiguous_hits += 1
            else:
                break

        # Block-alignment invariant: ``hit_tokens`` is a whole multiple of
        # block_size because contiguous_hits is whole blocks. vLLM's
        # ``num_computed_tokens`` is also block-aligned (the native
        # prefix cache returns whole-block counts).
        hit_tokens = contiguous_hits * self._block_size

        # Defensive clamp. In theory ``hit_tokens >= num_computed_tokens``
        # because both walk the same prefix from index 0 and Lethe is
        # usually a superset of vLLM's local prefix cache. In practice
        # the two caches can diverge (Lethe may evict before vLLM does,
        # or vice versa), and a negative return would corrupt scheduler
        # state — vLLM stores the value into ``num_external_computed_
        # tokens`` and feeds it into a sum (scheduler.py:642). Clamp to
        # 0 and continue; tests/correctness/test_connector_scheduler.py::
        # test_get_num_clamps_negative forces the divergent case.
        new_tokens = max(0, hit_tokens - num_computed_tokens)
        return (new_tokens, False)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        raise NotImplementedError("W1.3")

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        # W1.3 returns a populated ``LetheConnectorMetadata`` and resets
        # ``self._requests_need_load`` at the end (see ``base.py:512`` —
        # "calling this function will reset the state of the connector").
        raise NotImplementedError("W1.3")
