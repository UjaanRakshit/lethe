"""Integration: failover correctness."""
from __future__ import annotations
import pytest


@pytest.mark.skip("requires docker-compose cluster")
def test_single_node_failure_recovers_within_budget():
    # 1) start 3-node cluster
    # 2) insert N blocks; verify each is on 2 nodes
    # 3) kill one node
    # 4) within RECOVERY_BUDGET_MS, every block must be on 2 alive nodes
    pass


@pytest.mark.skip
def test_partition_does_not_lose_acked_writes():
    pass
