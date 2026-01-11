"""gRPC client for the Lethe cache server.

Thin wrapper over the generated lethe_pb2 / lethe_pb2_grpc stubs
(produced by `scripts/build.sh` or `python -m grpc_tools.protoc`).

The wire protocol is proto/lethe.proto. This client targets:
  * Lookup       — metadata-only check for cache presence.
  * Insert       — push KV blocks into the cache.
  * Fetch        — pull a single block's bytes (W1 roundtrip path).
  * StreamBlocks — bulk push of blocks (used by W4+ replication).

W3+ multi-node: pass `peers=[(node_id, address), ...]` and the client
mirrors the server-side hash ring to pre-route each block. W1 uses
`primary_address` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import grpc

from . import lethe_pb2, lethe_pb2_grpc

logger = logging.getLogger(__name__)


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
    """Synchronous gRPC client. Use one instance per process; gRPC
    channels are themselves thread-safe.
    """

    def __init__(
        self,
        primary_address: str,
        peers: Optional[list[tuple[str, str]]] = None,
        timeout_seconds: float = 5.0,
    ):
        self.primary_address = primary_address
        self.peers = dict(peers or [])
        self.timeout_seconds = timeout_seconds
        self._channels: dict[str, grpc.Channel] = {}
        self._stubs: dict[str, lethe_pb2_grpc.LetheCacheStub] = {}

    # ---- Channel management ----------------------------------------------

    def _stub(self, address: Optional[str] = None) -> lethe_pb2_grpc.LetheCacheStub:
        addr = address or self.primary_address
        if addr not in self._stubs:
            ch = grpc.insecure_channel(addr)
            self._channels[addr] = ch
            self._stubs[addr] = lethe_pb2_grpc.LetheCacheStub(ch)
        return self._stubs[addr]

    def close(self) -> None:
        for ch in self._channels.values():
            ch.close()
        self._channels.clear()
        self._stubs.clear()

    def __enter__(self) -> "LetheClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ---- Public API ------------------------------------------------------

    def lookup(
        self,
        block_ids: Iterable[BlockId],
        request_id: str = "",
        requesting_node: str = "client",
        address: Optional[str] = None,
    ) -> LookupResult:
        req = lethe_pb2.LookupRequest(
            block_ids=[_to_proto_block_id(b) for b in block_ids],
            request_id=request_id,
            requesting_node=requesting_node,
        )
        resp = self._stub(address).Lookup(req, timeout=self.timeout_seconds)
        return LookupResult(
            hits=[
                LookupHit(
                    block_id=_from_proto_block_id(h.id),
                    source_node=h.source_node,
                    tier=h.tier,
                )
                for h in resp.hits
            ],
            misses=[_from_proto_block_id(m) for m in resp.misses],
        )

    def insert(
        self,
        blocks: Iterable[tuple[BlockId, bytes]],
        request_id: str = "",
        source_node: str = "client",
        tier_hint: int = 1,
        address: Optional[str] = None,
    ) -> int:
        """Insert prefill-produced KV blocks. Returns the count accepted."""
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
        resp = self._stub(address).Insert(req, timeout=self.timeout_seconds)
        return int(resp.accepted_count)

    def fetch(
        self,
        block_id: BlockId,
        requesting_node: str = "client",
        address: Optional[str] = None,
    ) -> Optional[bytes]:
        """Pull a single block's bytes. Returns None on miss.

        Pre-RDMA path (W1); for large bulk transfers the server uses
        StreamBlocks instead. Roundtrip semantics: the bytes returned
        here are an immutable copy — the cache's local span is never
        exposed across the wire.
        """
        req = lethe_pb2.FetchRequest(
            id=_to_proto_block_id(block_id),
            requesting_node=requesting_node,
        )
        resp = self._stub(address).Fetch(req, timeout=self.timeout_seconds)
        if not resp.found:
            return None
        return bytes(resp.kv_data)
