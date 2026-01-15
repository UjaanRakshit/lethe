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

import hashlib
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from .client import BlockId, LetheClient
from .routing import chained_block_hash

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor (de)serialization helpers
# ---------------------------------------------------------------------------

# torch dtype ↔ numpy dtype mapping for the dtypes we'll actually see in
# KV cache (Gemma-3-1B is fp16 by default; bf16 added because Llama
# defaults to bf16 and may surface in W2+ benchmarks).
_TORCH_TO_NUMPY: dict[torch.dtype, np.dtype] = {
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    # bf16 has no native numpy dtype; we view as uint16 for byte-roundtrip
    # and reconstruct with torch.from_numpy + .view(torch.bfloat16).
    torch.bfloat16: np.dtype(np.uint16),
}


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a (potentially-GPU) tensor to raw IEEE-754 bytes.

    Synchronous on purpose: ``save_kv_layer`` must capture the bytes
    BEFORE returning, because the paged-KV buffer can be overwritten
    by the next layer's forward — and the connector's executor would
    otherwise race against that overwrite.
    """
    if t.dtype is torch.bfloat16:
        # bf16 → uint16 view → numpy → bytes
        return t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
    return t.detach().cpu().contiguous().numpy().tobytes()


def _bytes_to_tensor(
    payload: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Inverse of ``_tensor_to_bytes``. The caller supplies the
    expected shape/dtype/device — we don't trust wire bytes alone.
    """
    np_dtype = _TORCH_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(
            f"Lethe: unsupported KV dtype {dtype}; extend _TORCH_TO_NUMPY"
        )
    expected_bytes = int(np.prod(shape)) * np_dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Lethe: KV payload size mismatch — got {len(payload)} bytes, "
            f"expected {expected_bytes} for shape={shape} dtype={dtype}. "
            f"This usually means the model config drifted between save and "
            f"load runs, or a layer's KV shape changed."
        )
    arr = np.frombuffer(payload, dtype=np_dtype).reshape(shape)
    # np.frombuffer returns a read-only view; copy so torch can own it.
    t = torch.from_numpy(arr.copy())
    if dtype is torch.bfloat16:
        t = t.view(torch.bfloat16)
    return t.to(device)


def _layer_id_for(layer_name: str) -> int:
    """Deterministic uint32 derived from layer_name. Stable across
    processes (unlike Python's randomized hash()), so a save in run B
    and a load in run C use the same BlockId.layer value.
    """
    return int.from_bytes(
        hashlib.sha256(layer_name.encode()).digest()[:4], "little"
    )


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

        # ---- Worker-side state ----------------------------------------
        #
        # Concurrency primitive: a single ThreadPoolExecutor for both
        # background fetches (load path) and background inserts (save
        # path). Rationale:
        #
        #   - LetheClient is sync-shaped (grpcio sync stubs). Wrapping
        #     it in asyncio would force every connector caller to be
        #     async-aware — out of proportion for W1.
        #   - threading is the minimum-surface-area async shim. grpc
        #     Channels are thread-safe, so worker threads can issue
        #     concurrent RPCs against the shared client.
        #   - max_workers=4 leaves CPU headroom for the model forward;
        #     small-block per-RPC means we're more latency-bound than
        #     throughput-bound, so a larger pool would help marginally
        #     at the cost of more thread overhead.
        #   - W6 RDMA generalizes via the same Future.result()
        #     contract, so the executor stays even when RDMA replaces
        #     gRPC underneath.
        self._executor: ThreadPoolExecutor | None = None
        if role is KVConnectorRole.WORKER:
            self._executor = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="lethe-worker"
            )

        # In-flight load futures keyed by (request_id, layer_id, block_id_index)
        # → Future[bytes]. Populated by start_load_kv, drained by
        # wait_for_layer_load. Lock guards the dict; futures are
        # individually completed by executor threads.
        self._inflight_loads: dict[tuple[str, int, int], Future] = {}
        self._inflight_loads_lock = threading.Lock()

        # In-flight save futures. List, not dict — we don't need
        # individual lookup, just a batch drain in wait_for_save.
        self._inflight_saves: list[Future] = []
        self._inflight_saves_lock = threading.Lock()

        # KV tensor shape pinning. Captured on the first save_kv_layer
        # call per (layer_name); asserted on every subsequent save and
        # on every load. If a save and load see different shapes for
        # the same layer, that is a real bug (model config drift,
        # tensor-parallelism mismatch, etc.) and we surface it loudly
        # rather than silently reshape.
        # Key: layer_name → (block_shape, torch.dtype).
        self._layer_shape_pin: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._layer_shape_pin_lock = threading.Lock()

        logger.info(
            "LetheCacheConnector init role=%s lethe=%s block_size=%d model_id=%d",
            role.name,
            self._lethe_address,
            self._block_size,
            self._model_id,
        )

    def __del__(self):
        # Best-effort executor shutdown. Avoid raising in __del__ even
        # if attributes don't exist yet (constructor exception path).
        try:
            ex = self.__dict__.get("_executor")
            if ex is not None:
                ex.shutdown(wait=False)
        except Exception:
            pass

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
        """Capture this layer's KV bytes for the store-eligible blocks
        recorded by ``build_connector_meta``, async-push to Lethe.

        Sync part: read the connector metadata, slice the per-block
        tensor out of ``kv_layer``, move bytes to CPU. Done in the
        calling thread because ``kv_layer`` may be overwritten by the
        next layer's forward.

        Async part: serialize bytes + Insert RPC. Submitted to the
        executor; ``wait_for_save`` blocks on completion.
        """
        if self._executor is None:
            # Non-worker role shouldn't be called here; bug if it is.
            raise RuntimeError(
                "save_kv_layer called on non-WORKER role connector"
            )

        try:
            metadata = self._get_connector_metadata()
        except AssertionError:
            # No metadata bound for this forward — nothing to save.
            return
        if not isinstance(metadata, LetheConnectorMetadata):
            return
        if not metadata.stores:
            return

        # Layer shape comes from kv_layer. Reference layout (non-MLA,
        # non-Triton): (2, num_pages, page_size, kv_dim_per_token)
        # where the leading 2 is K|V and page_size == block_size.
        if kv_layer.ndim != 4 or kv_layer.shape[0] != 2:
            raise RuntimeError(
                f"Lethe: unexpected kv_layer shape {tuple(kv_layer.shape)} "
                f"for layer {layer_name!r}; expected (2, num_pages, "
                f"page_size, kv_dim). MLA / Triton layouts are not "
                f"handled in W1; surface this and triage rather than "
                f"silently coercing."
            )
        page_size = kv_layer.shape[2]
        if page_size != metadata.block_size:
            raise RuntimeError(
                f"Lethe: kv_layer page_size={page_size} != "
                f"metadata.block_size={metadata.block_size} for layer "
                f"{layer_name!r}. Block-hash chain would diverge from "
                f"vLLM's block boundaries; refusing to save."
            )

        # Per-block expected shape: (2, page_size, kv_dim).
        per_block_shape = (2, page_size, kv_layer.shape[3])
        per_block_dtype = kv_layer.dtype

        # Pin shape+dtype for this layer on first sight; assert on
        # subsequent saves. A mismatch here is a real bug worth a
        # loud error rather than a silent reshape.
        with self._layer_shape_pin_lock:
            pinned = self._layer_shape_pin.get(layer_name)
            if pinned is None:
                self._layer_shape_pin[layer_name] = (per_block_shape, per_block_dtype)
                logger.info(
                    "Lethe: pinned layer %r shape=%s dtype=%s",
                    layer_name, per_block_shape, per_block_dtype,
                )
            elif pinned != (per_block_shape, per_block_dtype):
                raise RuntimeError(
                    f"Lethe: layer {layer_name!r} KV shape drifted from "
                    f"{pinned} to {(per_block_shape, per_block_dtype)} "
                    f"between save calls. Model config change?"
                )

        layer_id = _layer_id_for(layer_name)
        client = self._ensure_client()

        # Capture per-block bytes synchronously, then submit Insert
        # batches asynchronously.
        insert_batch: list[tuple[BlockId, bytes]] = []
        for payload in metadata.stores:
            for spec in payload.blocks:
                vllm_blk = spec.vllm_block_id
                if vllm_blk >= kv_layer.shape[1]:
                    # vLLM allocated a block ID outside this layer's pool
                    # (shouldn't happen at steady state). Skip rather
                    # than crash; surface via logger so it's visible.
                    logger.warning(
                        "Lethe: vllm_block_id=%d out of range "
                        "(num_pages=%d) for layer %r request=%s; skipping",
                        vllm_blk, kv_layer.shape[1], layer_name,
                        payload.request_id,
                    )
                    continue
                # Slice: kv_layer[:, vllm_blk, :, :] → (2, page_size, kv_dim)
                block_tensor = kv_layer[:, vllm_blk, :, :]
                block_bytes = _tensor_to_bytes(block_tensor)
                block_id = BlockId(
                    hash=spec.chained_hash,
                    layer=layer_id,
                    head_group=0,
                    model_id=self._model_id,
                )
                insert_batch.append((block_id, block_bytes))

        if not insert_batch:
            return

        # Submit one Insert RPC per layer-step (batched across all
        # store-eligible blocks for all scheduled requests in this step).
        request_id = f"save-{layer_name}-{id(metadata):x}"
        future = self._executor.submit(
            client.insert,
            insert_batch,
            request_id,
            "lethe-worker",
        )
        with self._inflight_saves_lock:
            self._inflight_saves.append(future)

    def wait_for_save(self) -> None:
        """Drain all pending save Inserts submitted this forward.

        Called by vLLM at forward-context exit. We block until every
        future returned by ``save_kv_layer``'s executor.submit completes.
        Any exception from a save Insert is logged but not re-raised:
        a cache write failure must not stall the engine.
        """
        with self._inflight_saves_lock:
            futures = self._inflight_saves
            self._inflight_saves = []
        for fut in futures:
            try:
                fut.result(timeout=30.0)
            except Exception as e:  # noqa: BLE001 — grpc raises a wide tree
                logger.warning(
                    "Lethe: save Insert failed (not re-raised): %s",
                    e,
                )

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
        """Record a request's pending external load so ``build_connector_meta``
        can package it into the worker-bound metadata.

        Threading: same single-threaded scheduler contract as
        ``get_num_new_matched_tokens``.
        """
        if num_external_tokens <= 0:
            return

        # KVCacheBlocks.get_block_ids() returns ``tuple[list[int], ...]``
        # — outer tuple is per KV-cache-group. W1 single-group assumption;
        # heterogeneous block sizes per group are an HMA concern
        # (SupportsHMA mixin) and out of W1 scope.
        block_id_groups = blocks.get_block_ids()
        if not block_id_groups or not block_id_groups[0]:
            return

        n_blocks = num_external_tokens // self._block_size
        if n_blocks == 0:
            # vLLM only allocates full blocks; getting num_external_tokens
            # smaller than a block here would mean a config drift.
            return

        # Defensive: bound by what was actually allocated (vLLM may pre-
        # allocate fewer slots than we asked for under capacity pressure).
        n_blocks = min(n_blocks, len(block_id_groups[0]))

        self._requests_need_load[request.request_id] = {
            "n_blocks": n_blocks,
            "allocated_block_ids": list(block_id_groups[0][:n_blocks]),
        }

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Package this step's load/store work into the metadata that
        will ride to the worker side via ``bind_connector_metadata``.

        Per the base.py docstring at line 512:
            "calling this function will reset the state of the
             connector"
        We honor that by ``pop``-ing entries out of ``_requests_need_load``
        as we consume them — a second call on the same scheduler_output
        sees no pending loads.

        W1 store policy: every newly-scheduled request stores every
        whole-block of its prompt that ISN'T being loaded. Cold-prefilled
        requests store everything; partially-warm requests store only
        the new (post-load) blocks. W4+ may add per-block hit-skip and
        replication wave-shaping.
        """
        meta = LetheConnectorMetadata(block_size=self._block_size)

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            if not token_ids:
                continue
            if not new_req.block_ids or not new_req.block_ids[0]:
                continue
            allocated = new_req.block_ids[0]

            # How many blocks this request will load from Lethe (0 if
            # cold, >0 if update_state_after_alloc recorded it).
            state = self._requests_need_load.pop(new_req.req_id, None)
            n_load = int(state["n_blocks"]) if state else 0

            # Whole prefix blocks that have a paged-KV slot allocated.
            n_whole_blocks = min(
                len(token_ids) // self._block_size,
                len(allocated),
            )
            if n_whole_blocks == 0:
                continue

            load_specs: list[LetheBlockSpec] = []
            store_specs: list[LetheBlockSpec] = []
            running = b"\x00" * 32
            for i in range(n_whole_blocks):
                start = i * self._block_size
                block_tokens = token_ids[start : start + self._block_size]
                running = chained_block_hash(running, block_tokens)
                spec = LetheBlockSpec(
                    chained_hash=running,
                    vllm_block_id=int(allocated[i]),
                    is_hit=(i < n_load),
                )
                if i < n_load:
                    load_specs.append(spec)
                else:
                    store_specs.append(spec)

            if load_specs:
                meta.loads.append(
                    LetheRequestPayload(
                        request_id=new_req.req_id,
                        blocks=load_specs,
                    )
                )
            if store_specs:
                meta.stores.append(
                    LetheRequestPayload(
                        request_id=new_req.req_id,
                        blocks=store_specs,
                    )
                )

        return meta
