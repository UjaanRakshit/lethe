"""Prefix-chained block hashing and client-side routing helpers.

The hash chain is the entire reason prefix-aware routing works for free:
   block_hash[i] = H( block_hash[i-1] || tokens_in_block_i )

Two sequences sharing the first k tokens produce identical block_hash[0..k]
and therefore route to the same owner under consistent hashing.

This module mirrors the C++ Router so that the client can pre-route requests
without an extra RPC round-trip. The hash function must match exactly: BLAKE3
in 32-byte output mode.
"""

from __future__ import annotations

import struct
from typing import Iterable

try:
    import blake3
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "blake3 is required for cross-language hash compatibility with "
        "the C++ cache server — install with `pip install blake3`. "
        "The previous SHA-256 fallback was removed deliberately because "
        "the two languages must produce identical digests for every "
        "`(prev_hash, tokens)` and every `f\"{peer}#{vn}\"` ring key; "
        "a silent fallback would produce a working but incorrect cluster."
    ) from e

_HASH = blake3.blake3


def chained_block_hash(prev_hash: bytes, block_tokens: Iterable[int]) -> bytes:
    """Compute H(prev_hash || tokens). Returns 32 bytes."""
    h = _HASH(prev_hash)
    buf = b"".join(struct.pack("<I", t) for t in block_tokens)
    h.update(buf)
    return h.digest()


def prefix_block_ids(
    token_ids: list[int],
    block_size: int,
    initial_hash: bytes = b"\x00" * 32,
) -> list[bytes]:
    """Compute the chained block hashes for a tokenized sequence."""
    out: list[bytes] = []
    running = initial_hash
    for start in range(0, len(token_ids), block_size):
        block = token_ids[start : start + block_size]
        running = chained_block_hash(running, block)
        out.append(running)
    return out


# ---- Consistent-hash ring (client mirror of Router) -----------------------

class HashRing:
    """64-bit consistent hash ring with virtual nodes.

    Mirrors cache_server/include/lethe/routing.hpp. Construct once with the
    current peer list; rebuild on membership change.
    """

    def __init__(self, peers: list[str], vnodes_per_peer: int = 128):
        self.vnodes_per_peer = vnodes_per_peer
        self._ring: list[tuple[int, str]] = []
        self.set_peers(peers)

    def set_peers(self, peers: list[str]) -> None:
        ring: list[tuple[int, str]] = []
        for peer in peers:
            for vn in range(self.vnodes_per_peer):
                key = f"{peer}#{vn}".encode()
                # First 8 bytes of BLAKE3, little-endian, as uint64.
                h = _HASH(key).digest()
                ring.append((int.from_bytes(h[:8], "little"), peer))
        ring.sort()
        self._ring = ring

    def route(self, block_hash: bytes, n_replicas: int = 2) -> list[str]:
        """Return the primary + (n_replicas - 1) successor peers."""
        if not self._ring:
            return []
        target = int.from_bytes(block_hash[:8], "little")
        # Binary search for first ring entry with hash >= target.
        lo, hi = 0, len(self._ring)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._ring[mid][0] < target:
                lo = mid + 1
            else:
                hi = mid
        idx = lo % len(self._ring)
        out: list[str] = []
        seen: set[str] = set()
        i = idx
        while len(out) < n_replicas and len(seen) < self.vnodes_per_peer * 32:
            peer = self._ring[i][1]
            if peer not in seen:
                out.append(peer)
                seen.add(peer)
            i = (i + 1) % len(self._ring)
        return out
