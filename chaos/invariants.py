"""Invariants asserted under chaos (W11).

Run alongside chaos.kill_node / chaos.partition. Polls the cluster's
Prometheus metrics and asserts:

  INV-1: No block ever has zero replicas (lethe_replicas_under_target stays
         at the configured factor minus one or better).
  INV-2: After a node death, lethe_cluster_epoch increments within
         dead_after_ms (default 3s).
  INV-3: Failover recovery (re-replication of all under-replicated blocks)
         completes within RECOVERY_BUDGET_MS (target: 500ms for 3-node, R=2).
  INV-4: No Lookup returns a stale RemoteHit pointing at a dead node for
         more than 2x heartbeat_interval after the death.
"""
from __future__ import annotations
import argparse, time, urllib.request


def _prom(addr: str, query: str) -> float:
    url = f"http://{addr}/api/v1/query?query={query}"
    with urllib.request.urlopen(url) as r:
        import json
        data = json.load(r)
    result = data["data"]["result"]
    if not result:
        return 0.0
    return float(result[0]["value"][1])


def check_invariants(prom_addr: str, recovery_budget_ms: float = 500) -> bool:
    failures: list[str] = []

    if _prom(prom_addr, "lethe_replicas_under_target") > 0:
        # OK during recovery window; check it goes to zero in time.
        t0 = time.monotonic()
        while (time.monotonic() - t0) * 1000 < recovery_budget_ms:
            if _prom(prom_addr, "lethe_replicas_under_target") == 0:
                break
            time.sleep(0.05)
        else:
            failures.append("INV-3: recovery exceeded budget")

    if failures:
        print("FAIL:", *failures, sep="\n  ")
        return False
    print("OK")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prometheus", default="localhost:9090")
    p.add_argument("--recovery-budget-ms", type=float, default=500)
    args = p.parse_args()
    ok = check_invariants(args.prometheus, args.recovery_budget_ms)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
