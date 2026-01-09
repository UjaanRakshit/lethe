"""Unit: consistent-hash router (W3)."""
from __future__ import annotations

from lethe_client.routing import HashRing


def test_route_returns_n_distinct_peers():
    ring = HashRing(["a", "b", "c"], vnodes_per_peer=64)
    out = ring.route(b"\x01" * 32, n_replicas=2)
    assert len(out) == 2
    assert len(set(out)) == 2


def test_same_block_routes_consistently():
    ring = HashRing(["a", "b", "c"], vnodes_per_peer=64)
    h = b"\x42" * 32
    assert ring.route(h, 2) == ring.route(h, 2)


def test_membership_change_minimal_disruption():
    """Removing one peer should only move ~1/N of keys."""
    import random
    ring = HashRing(["a", "b", "c", "d"], vnodes_per_peer=128)
    keys = [random.randbytes(32) for _ in range(1000)]
    before = {k: ring.route(k, 1)[0] for k in keys}
    ring.set_peers(["a", "b", "c"])
    after = {k: ring.route(k, 1)[0] for k in keys}

    moved = sum(1 for k in keys if before[k] != after[k])
    # Expect roughly 25% to move (the share owned by 'd'). Allow generous
    # slack for variance with vnodes_per_peer=128.
    assert 0.15 < moved / len(keys) < 0.40
