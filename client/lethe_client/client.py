"""gRPC client for the Lethe cache server.

Thin wrapper over the generated lethe_pb2 / lethe_pb2_grpc stubs. RPCs:
Lookup (presence check), Insert (push blocks), Fetch (pull one block's
bytes), StreamBlocks (bulk push, used by replication).

Single-node mode: pass `primary_address` only; every RPC hits that node.

Multi-node mode: pass `peers=[(node_id, address), ...]`. The client builds
a `HashRing` (bit-compatible with the C++ Router) and routes every Lookup
per-block to its primary owner, batching block_ids that share a primary
into one RPC. Inserts hit the local node only - the server-side Replicator
handles cross-node pushes. On a RemoteHit response, the client
transparently Fetches from source_node and stitches the bytes into the
result.

Failure semantics (never block scheduling on cache liveness):
transient gRPC errors (UNAVAILABLE / DEADLINE_EXCEEDED) retry up to 3
times with 50 ms × 2^attempt backoff; other status codes surface as Miss.
A cross-node Fetch failure reclassifies the block as Miss. lookup/fetch
never raise grpc.RpcError - they return a LookupResult (possibly all
misses) or bytes-or-None.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import grpc

from . import lethe_pb2, lethe_pb2_grpc
from .routing import HashRing

logger = logging.getLogger(__name__)


# Retry policy: 3 attempts, 50 ms × 2^n exponential backoff, retrying
# only UNAVAILABLE / DEADLINE_EXCEEDED. Other RPC failures fall through
# as Miss / None rather than raising into the connector's stack.
_RETRY_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
}
_MAX_ATTEMPTS = 3
_BACKOFF_BASE_SECONDS = 0.050


@dataclass(frozen=True)
class BlockId:
    """Mirror of the C++ BlockId. Hash is 32 bytes."""

    hash: bytes
    layer: int = 0
    head_group: int = 0
    model_id: int = 0


@dataclass
class LookupHit:
    block_id: BlockId
    source_node: str
    tier: int


@dataclass
class LookupResult:
    hits: list[LookupHit] = field(default_factory=list)
    misses: list[BlockId] = field(default_factory=list)
    # Transparently-fetched RemoteHit bytes, keyed by hash. None entries
    # are blocks that were RemoteHit but whose follow-up Fetch failed
    # (treat as Miss).
    fetched: dict[bytes, Optional[bytes]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = len(self.hits) + len(self.misses)
        return (len(self.hits) / total) if total else 0.0


def _to_proto_block_id(bid: BlockId) -> lethe_pb2.BlockId:
    return lethe_pb2.BlockId(
        hash=bid.hash,
        layer=bid.layer,
        head_group=bid.head_group,
        model_id=bid.model_id,
    )


def _from_proto_block_id(pb: lethe_pb2.BlockId) -> BlockId:
    return BlockId(
        hash=bytes(pb.hash),
        layer=pb.layer,
        head_group=pb.head_group,
        model_id=pb.model_id,
    )


class LetheClient:
    """Synchronous gRPC client. Thread-safe for concurrent Lookup /
    Insert / Fetch calls (gRPC Channels are themselves thread-safe).

    Single-node usage:
        LetheClient("127.0.0.1:50051")

    Multi-node usage:
        LetheClient(
            primary_address="127.0.0.1:50051",  # any reachable node
            peers=[("node0", "127.0.0.1:50051"),
                   ("node1", "127.0.0.1:50052"),
                   ("node2", "127.0.0.1:50053")],
        )

    When ``peers`` is non-empty, Lookup routes per-block via a local
    ``HashRing`` that mirrors the server's Router. ``primary_address``
    is still used for Insert (the server-side Replicator handles
    cross-node fanout) and as a fallback when routing produces no
    primary (empty peer set or unknown peer).
    """

    def __init__(
        self,
        primary_address: str,
        peers: Optional[list[tuple[str, str]]] = None,
        timeout_seconds: float = 5.0,
    ):
        self.primary_address = primary_address
        self.peers: dict[str, str] = dict(peers or [])
        self.timeout_seconds = timeout_seconds
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, lethe_pb2_grpc.LetheCacheStub] = {}
        # HashRing built from the peer set. Empty peers → empty ring
        # → routing falls back to primary_address. The ring is
        # bit-compatible with the C++ Router.
        self._ring: Optional[HashRing] = None
        if self.peers:
            self._ring = HashRing(list(self.peers.keys()))

    # ---- Channel / stub management ---------------------------------------

    def _stub(self, address: Optional[str] = None) -> lethe_pb2_grpc.LetheCacheStub:
        addr = address or self.primary_address
        if addr not in self._stubs:
            ch = grpc.insecure_channel(addr)
            self._channels[addr] = ch
            self._stubs[addr] = lethe_pb2_grpc.LetheCacheStub(ch)
        return self._stubs[addr]

    def _stub_for_node(self, node_id: str) -> Optional[lethe_pb2_grpc.LetheCacheStub]:
        """Stub for a node_id, or None if we have no address for it."""
        addr = self.peers.get(node_id)
        if not addr:
            return None
        return self._stub(addr)

    def close(self) -> None:
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()
        self._stubs.clear()

    def __enter__(self) -> "LetheClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ---- Retry helper ----------------------------------------------------

    def _call_with_retry(self, rpc_callable):
        """Run an RPC closure with the retry policy. Returns the response
        on success, None on terminal failure (the caller decides how to
        surface it - Miss for Lookup, None for Fetch).

        ``rpc_callable`` is a no-arg lambda that issues one RPC; it may
        raise grpc.RpcError, which the helper catches and either retries
        (UNAVAILABLE / DEADLINE_EXCEEDED) or gives up on.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                return rpc_callable()
            except grpc.RpcError as e:
                last_exc = e
                code = e.code() if hasattr(e, "code") else None
                if code not in _RETRY_CODES:
                    logger.warning(
                        "Lethe RPC failed (non-retryable code=%s): %s",
                        code, e.details() if hasattr(e, "details") else e,
                    )
                    return None
                if attempt + 1 < _MAX_ATTEMPTS:
                    sleep = _BACKOFF_BASE_SECONDS * (2 ** attempt)
                    logger.debug(
                        "Lethe RPC retry %d/%d on code=%s, sleep=%.3fs",
                        attempt + 1, _MAX_ATTEMPTS - 1, code, sleep,
                    )
                    time.sleep(sleep)
        logger.warning(
            "Lethe RPC failed after %d attempts: %s",
            _MAX_ATTEMPTS, last_exc,
        )
        return None

    # ---- Public API ------------------------------------------------------

    def lookup(
        self,
        block_ids: Iterable[BlockId],
        request_id: str = "",
        requesting_node: str = "client",
        address: Optional[str] = None,
    ) -> LookupResult:
        """Resolve block IDs against the cluster.

        In multi-node mode, partitions ``block_ids`` by primary owner
        via the HashRing and issues ONE RPC per distinct primary
        (not one RPC per block). Per-block batching is mandatory:
        long-context Lookups against R=2 with 128 vnodes will see
        ~3 distinct primaries on a 3-node cluster, so the wire cost
        stays O(number-of-nodes), not O(number-of-blocks).

        On RemoteHit response from a node, transparently issue a
        Fetch RPC against the named source_node and stitch the bytes
        into ``LookupResult.fetched``. Fetch failures degrade to
        Miss for that block.

        Never raises. RPC failure for an entire primary group →
        every block in that group is reported as Miss.
        """
        ids = list(block_ids)
        result = LookupResult()
        if not ids:
            return result

        # Partition by primary. If no ring is configured OR a block's
        # primary isn't in the peer map, route to primary_address.
        groups: dict[Optional[str], list[BlockId]] = {}
        for bid in ids:
            primary_node: Optional[str] = None
            if self._ring is not None:
                routed = self._ring.route(bid.hash, n_replicas=1)
                if routed and routed[0] in self.peers:
                    primary_node = routed[0]
            groups.setdefault(primary_node, []).append(bid)

        # One RPC per distinct primary.
        for primary_node, batch in groups.items():
            stub = (
                self._stub_for_node(primary_node)
                if primary_node is not None
                else self._stub(address)
            )
            if stub is None:
                # No address for the resolved primary; treat the whole
                # batch as Miss. Surface via logger so a stale peer
                # configuration is visible.
                logger.warning(
                    "Lethe lookup: no peer address for node_id=%r; "
                    "%d blocks routed to Miss", primary_node, len(batch),
                )
                for b in batch:
                    result.misses.append(b)
                continue

            req = lethe_pb2.LookupRequest(
                block_ids=[_to_proto_block_id(b) for b in batch],
                request_id=request_id,
                requesting_node=requesting_node,
            )
            resp = self._call_with_retry(
                lambda s=stub, r=req: s.Lookup(r, timeout=self.timeout_seconds)
            )
            if resp is None:
                # Whole batch failed. Surface as Miss; the caller
                # decides whether to retry against a different node.
                for b in batch:
                    result.misses.append(b)
                continue

            # Local-style hits land directly; remote hits trigger a
            # transparent Fetch.
            for h in resp.hits:
                hit = LookupHit(
                    block_id=_from_proto_block_id(h.id),
                    source_node=h.source_node,
                    tier=h.tier,
                )
                result.hits.append(hit)
                # If the server returned a remote hit, fetch the bytes
                # from source_node so the caller doesn't have to. The
                # server returns LocalHit when it has the bytes locally
                # (including after read-repair); RemoteHit only surfaces
                # when it can't serve the bytes itself.
                if h.source_node and h.source_node != self._local_label_for(stub):
                    self._fetch_into_result(hit.block_id, h.source_node, result)
            for m in resp.misses:
                result.misses.append(_from_proto_block_id(m))

        return result

    def insert(
        self,
        blocks: Iterable[tuple[BlockId, bytes]],
        request_id: str = "",
        source_node: str = "client",
        tier_hint: int = 1,
        address: Optional[str] = None,
    ) -> int:
        """Insert prefill-produced KV blocks. Returns the count accepted.

        Multi-node: Insert hits ONE node (``address`` or
        ``primary_address``); the server-side Replicator pushes to
        the R-1 successors on the ring. We don't route Insert by
        block on the client side - the wire cost would force the
        client to know about every primary, and the server is more
        efficient at the fanout.
        """
        req = lethe_pb2.InsertRequest(
            blocks=[
                lethe_pb2.InsertRequest.Block(
                    id=_to_proto_block_id(bid),
                    kv_data=payload,
                    tier_hint=tier_hint,
                )
                for bid, payload in blocks
            ],
            request_id=request_id,
            source_node=source_node,
        )
        stub = self._stub(address)
        resp = self._call_with_retry(
            lambda: stub.Insert(req, timeout=self.timeout_seconds)
        )
        if resp is None:
            return 0
        return int(resp.accepted_count)

    def fetch(
        self,
        block_id: BlockId,
        requesting_node: str = "client",
        address: Optional[str] = None,
    ) -> Optional[bytes]:
        """Pull a single block's bytes. Returns None on miss or
        terminal RPC failure. Roundtrip semantics: bytes are an
        immutable copy - the cache's local span never crosses the
        wire.
        """
        req = lethe_pb2.FetchRequest(
            id=_to_proto_block_id(block_id),
            requesting_node=requesting_node,
        )
        stub = self._stub(address)
        resp = self._call_with_retry(
            lambda: stub.Fetch(req, timeout=self.timeout_seconds)
        )
        if resp is None or not resp.found:
            return None
        return bytes(resp.kv_data)

    # ---- Internals -------------------------------------------------------

    def _local_label_for(self, _stub) -> str:
        # The client doesn't know its own node identity; this empty
        # sentinel lets the RemoteHit-triggered Fetch path avoid spinning
        # on a hit whose source_node matches the queried node. The server
        # already returns LocalHit in that case, so this branch is purely
        # defensive.
        return ""

    def _fetch_into_result(
        self,
        block_id: BlockId,
        source_node: str,
        result: LookupResult,
    ) -> None:
        """Transparent fetch: dispatch a Fetch to source_node and
        record the bytes (or None for failure) in result.fetched."""
        if source_node not in self.peers:
            # Server returned a source_node we don't have an address
            # for (stale peer config on this client). Record a None.
            logger.warning(
                "Lethe lookup: RemoteHit source_node=%r unknown to client; "
                "block reclassified as Miss", source_node,
            )
            result.fetched[block_id.hash] = None
            return
        addr = self.peers[source_node]
        bytes_or_none = self.fetch(block_id, address=addr)
        result.fetched[block_id.hash] = bytes_or_none
