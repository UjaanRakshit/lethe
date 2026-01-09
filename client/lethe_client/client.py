"""gRPC client for the Lethe cache server.

This is a thin wrapper around the generated lethe_pb2/lethe_pb2_grpc stubs
(produced from proto/lethe.proto by `python -m grpc_tools.protoc`).

W1: Lookup, Insert work against a single node.
W3: client picks the primary owner for each block via the local Router mirror.
W4: client transparently retries against replicas on miss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

# Generated stubs — produced at build time:
#   python -m grpc_tools.protoc \
#     -Iproto --python_out=client/lethe_client \
#     --grpc_python_out=client/lethe_client proto/lethe.proto
#
# import lethe_pb2
# import lethe_pb2_grpc

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
    hits: list[LookupHit]
    misses: list[BlockId]

    @property
    def hit_rate(self) -> float:
        total = len(self.hits) + len(self.misses)
        return (len(self.hits) / total) if total else 0.0


class LetheClient:
    """Synchronous gRPC client. Use one instance per process; methods are
    thread-safe insofar as gRPC channels are.

    For multi-node clusters (W3+), pass `peers=[(node_id, address), ...]`
    and the client will route each block to the owner before calling.
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
        # self._channels: dict[str, grpc.Channel] = {}
        # self._stubs: dict[str, lethe_pb2_grpc.LetheCacheStub] = {}

    # ----- Public API ------------------------------------------------------

    def lookup(
        self,
        block_ids: Iterable[BlockId],
        request_id: str,
        requesting_node: str = "client",
    ) -> LookupResult:
        """Resolve block IDs against the cluster.

        W1: hits primary_address only.
        W3: dispatches per-block to owner; merges responses.
        """
        # TODO(W1): wire to lethe_pb2_grpc.LetheCacheStub.Lookup.
        raise NotImplementedError

    def insert(
        self,
        blocks: Iterable[tuple[BlockId, bytes]],
        request_id: str,
        source_node: str = "client",
        tier_hint: int = 1,
    ) -> int:
        """Insert prefill-produced KV blocks. Returns count accepted."""
        # TODO(W1): wire to LetheCacheStub.Insert.
        raise NotImplementedError

    def fetch(
        self,
        block_ids: Iterable[BlockId],
        from_node: str,
    ) -> dict[BlockId, bytes]:
        """Stream blocks from a peer node (used after Lookup → RemoteHit)."""
        # TODO(W4): wire to bidi StreamBlocks.
        raise NotImplementedError

    def close(self) -> None:
        # for ch in self._channels.values(): ch.close()
        pass
