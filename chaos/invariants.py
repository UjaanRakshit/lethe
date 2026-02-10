"""Invariant verification under failure injection (W11).

This module drives a running 3-node Lethe cluster through five chaos
scenarios and asserts the cluster's safety + liveness invariants. It is
both a library (the `ClusterProbe` + scenario functions) and a CLI:

    python -m chaos.invariants --scenario sigkill
    python -m chaos.invariants --scenario all

WHY behavioral probing instead of Prometheus PromQL (the W0 stub's plan):
  * Prometheus scrapes every 5s (deploy/prometheus.yml). A 3.5s recovery
    budget cannot be measured at 5s resolution.
  * `lethe_replicas_under_target` is NOT a live "current under-replication"
    gauge — replication.cpp sets it to the *dispatched count* on the last
    membership change and never clears it, so "wait until it hits 0" never
    fires. (See docs/DECISIONS.md 2026-05-28.)
  * `lethe_failover_recovery_seconds` is declared but has no call site, so
    the histogram is always empty.
  The only metric we trust here is `lethe_cluster_epoch`, scraped directly
  off each node's /metrics (sub-second). Everything else is verified by
  *behavior*: insert a known corpus, then probe each surviving node with a
  local-only Fetch (the server's Fetch handler is FetchLocal — no peer
  recursion, so a hit means that node physically holds the bytes).

THE INVARIANTS
  INV-1  No data loss. Every corpus block is retrievable from >=1 surviving
         node at all times (R=2 means one death never zeroes a block).
  INV-2  Failure detected. A survivor's cluster_epoch increments within the
         detection budget (dead_after=3s + slack) of a death.
  INV-3  Failover converges. Every block returns to the replication target
         (R=2 across survivors) within the recovery budget.
  INV-4  No stale routing. After detection completes, no survivor returns a
         RemoteHit pointing at the dead node.
  INV-5  Load path stays alive. The ring-routed hit-rate (the W9-lesson
         metric) never collapses to zero through failure + recovery.
  INV-6  No corruption. Every byte a node serves for a BlockId matches the
         bytes inserted for that BlockId (content integrity).

REAL BUG vs EXPECTED TRANSIENT (the judgment this suite encodes):
  * Data loss, corruption, a load path stuck at zero, non-convergence, or a
    survivor routing to a dead node *after* detection => REAL BUG. Fail hard.
  * R=1 briefly, a hit-rate dip, divergent ring views during a partition,
    a node dying under heavy packet loss => EXPECTED. The invariant must
    tolerate these, not flag them.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from lethe_client.client import BlockId, LetheClient
from lethe_client.routing import HashRing

from chaos.kill_node import is_running, kill, revive
from chaos.packet_loss import clear as netem_clear
from chaos.packet_loss import loss as netem_loss
from chaos.partition import heal as partition_heal
from chaos.partition import partition as partition_inject

# --------------------------------------------------------------------------
# Cluster topology (host-side view: docker publishes node client ports to
# 50051/61/71 and the per-node /metrics endpoints to 9091/2/3).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    node_id: str
    container: str
    client_addr: str  # host:port for gRPC
    metrics_addr: str  # host:port for /metrics


TOPOLOGY: list[Node] = [
    Node("node0", "lethe-node0", "127.0.0.1:50051", "127.0.0.1:9091"),
    Node("node1", "lethe-node1", "127.0.0.1:50061", "127.0.0.1:9092"),
    Node("node2", "lethe-node2", "127.0.0.1:50071", "127.0.0.1:9093"),
]
BY_ID = {n.node_id: n for n in TOPOLOGY}

# --------------------------------------------------------------------------
# Budgets. These mirror CLAUDE.md's "Architecture spine": detection floor is
# dead_after=3000ms; the documented end-to-end recovery target is 3.5s
# (3s detect + 500ms re-replicate). We assert a HARD ceiling well above the
# target (clearly-broken => fail) and separately WARN if the 3.5s target was
# missed — chasing the exact 3.5s edge is flaky and CLAUDE.md explicitly
# warns against tightening to hit a number.
# --------------------------------------------------------------------------
REPLICATION_FACTOR = 2
HEARTBEAT_MS = 200
DEAD_AFTER_MS = 3000
DETECTION_BUDGET_MS = 4200  # epoch must bump within this of the kill

# Recovery budget, RESTATED honestly in W11.1 (see docs/weekly/W11_1.md and
# docs/DESIGN.md). The W0 spine's "3s detect + 500ms re-replicate = 3.5s" was
# only ever true at tiny working sets — re-replication time scales ~linearly
# with the working set (the push queue drains at a measured rate). Detection
# is ~3s regardless; re-replication adds the measured per-size cost. INV-3
# asserts full reconvergence within the size-appropriate budget below and
# FAILS only on genuine non-convergence.
RECOVERY_DETECT_MS = 3000              # dead_after, size-independent
# Per-block re-replication cost (measured, W11.1 curve) + a fixed slack. Used
# to size the budget for a given corpus: budget = detect + N * per_block + slack.
RECOVERY_PER_BLOCK_MS = 40             # ~25 blocks/s drain (docker bridge)
RECOVERY_SLACK_MS = 4000


def recovery_budget_ms(n_blocks: int) -> float:
    """Honest, size-aware recovery budget: detection + measured drain + slack."""
    return RECOVERY_DETECT_MS + n_blocks * RECOVERY_PER_BLOCK_MS + RECOVERY_SLACK_MS


INV4_SETTLE_MS = DEAD_AFTER_MS + 1200  # after this, no stale RemoteHit

# kBoundedScan in replication.cpp: re-replication dispatches at most this many
# blocks PER sweep tick. Before W11.1 a single pass ran and nothing
# re-triggered, so working sets > this stayed under-replicated forever
# (Finding B). W11.1 wired a periodic sweep that drains successive batches
# until the whole round is covered — completeness now holds at any size; the
# cap only bounds per-tick work. The LARGE scenario below deliberately exceeds
# it to prove the fix.
REPLICATION_SCAN_CAP = 256

CORPUS_SIZE = 48
LARGE_CORPUS_SIZE = 600  # > REPLICATION_SCAN_CAP: the Finding B regression test
PAYLOAD_LEN = 256


def make_corpus(tag: str, n: int = CORPUS_SIZE) -> dict[bytes, bytes]:
    """Deterministic id -> payload map. Payload is derived from the id so a
    single returned byte string is enough to detect corruption (no need to
    carry the original around — recompute and compare)."""
    corpus: dict[bytes, bytes] = {}
    for i in range(n):
        h = hashlib.sha256(f"{tag}-{i}".encode()).digest()  # 32 bytes
        payload = (hashlib.sha256(h).digest() * 8)[:PAYLOAD_LEN]
        corpus[h] = payload
    return corpus


# --------------------------------------------------------------------------
# Probe: the behavioral measurement surface.
# --------------------------------------------------------------------------


@dataclass
class ReplicaState:
    holders: dict[bytes, list[str]]  # block_hash -> node_ids holding correct bytes
    corrupt: dict[bytes, list[str]]  # block_hash -> node_ids serving WRONG bytes
    lost: list[bytes]  # blocks on zero alive nodes
    min_holders: int  # min replica count across the corpus


class ClusterProbe:
    def __init__(self, topology: list[Node]):
        self.topology = topology
        self._single: dict[str, LetheClient] = {}
        self._ring: Optional[LetheClient] = None

    # -- client management -------------------------------------------------
    def single(self, node: Node) -> LetheClient:
        """A ring-free client pinned to one node — every RPC hits that node
        directly (no peers => no HashRing => no routing). Lets us ask a
        single node 'do you physically hold this block'."""
        if node.client_addr not in self._single:
            self._single[node.client_addr] = LetheClient(
                primary_address=node.client_addr, timeout_seconds=2.0
            )
        return self._single[node.client_addr]

    def ring(self) -> LetheClient:
        """The realistic, production-shape client: all peers configured, so
        Lookup routes per-block to its primary via the mirror HashRing."""
        if self._ring is None:
            peers = [(n.node_id, n.client_addr) for n in self.topology]
            self._ring = LetheClient(
                primary_address=self.topology[0].client_addr,
                peers=peers,
                timeout_seconds=2.0,
            )
        return self._ring

    def close(self) -> None:
        for c in self._single.values():
            c.close()
        self._single.clear()
        if self._ring is not None:
            self._ring.close()
            self._ring = None

    # -- workload ----------------------------------------------------------
    def insert_corpus(self, corpus: dict[bytes, bytes]) -> int:
        """Load the corpus ROUTED: each block is inserted at its ring-primary
        so the server's ReplicateOut pushes it to the correct replica and the
        block lands at the full R=2.

        Inserting everything via one node (the W10 quick-load pattern) leaves
        ~1/3 of blocks at R=1: for blocks where that node is the ring *replica*
        rather than the primary, ReplicateOut pushes only to replicas-minus-self
        and there is no other successor to push to. ReplicateOut itself flags
        insert-to-non-primary as the 'by mistake' path. A chaos suite that
        asserts 'no data loss on any single death' needs a genuine R=2 baseline,
        so we route inserts to the primary (the system's intended usage) — this
        is a load-pattern choice, not a change to routing/replication."""
        ring = HashRing([n.node_id for n in self.topology])
        groups: dict[str, list[tuple[BlockId, bytes]]] = {}
        for h, p in corpus.items():
            routed = ring.route(h, n_replicas=1)
            primary = routed[0] if routed else self.topology[0].node_id
            groups.setdefault(primary, []).append((BlockId(hash=h), p))
        total = 0
        for node_id, blocks in groups.items():
            total += self.single(BY_ID[node_id]).insert(blocks, request_id="w11-load")
        return total

    def probe_replicas(
        self, corpus: dict[bytes, bytes], alive: list[Node]
    ) -> ReplicaState:
        """For every block, ask each alive node (local-only Fetch) whether it
        holds the correct bytes. Classifies present / corrupt / lost."""
        holders: dict[bytes, list[str]] = {}
        corrupt: dict[bytes, list[str]] = {}
        lost: list[bytes] = []
        for h, want in corpus.items():
            ok_nodes: list[str] = []
            bad_nodes: list[str] = []
            for n in alive:
                got = self.single(n).fetch(BlockId(hash=h))
                if got is None:
                    continue
                if got == want:
                    ok_nodes.append(n.node_id)
                else:
                    bad_nodes.append(n.node_id)
            holders[h] = ok_nodes
            if bad_nodes:
                corrupt[h] = bad_nodes
            if not ok_nodes:
                lost.append(h)
        min_h = min((len(v) for v in holders.values()), default=0)
        return ReplicaState(holders, corrupt, lost, min_h)

    def ring_hit_rate(self, corpus: dict[bytes, bytes]) -> float:
        """Ring-routed lookup of the whole corpus; returns hits/(hits+miss).
        This is exactly the lethe_requests_total{result=hit} surface — the
        load-bearing W9-lesson metric."""
        ids = [BlockId(hash=h) for h in corpus]
        r = self.ring().lookup(ids, request_id="w11-hr")
        return r.hit_rate

    def stale_remotehit_count(
        self, corpus: dict[bytes, bytes], survivor: Node, dead_id: str
    ) -> int:
        """Ask one survivor (ring-free) to resolve the corpus; count hits
        whose source_node is the dead node. After detection drops the dead
        node from the survivor's ring, this must be zero."""
        ids = [BlockId(hash=h) for h in corpus]
        r = self.single(survivor).lookup(ids, request_id="w11-stale")
        return sum(1 for hit in r.hits if hit.source_node == dead_id)

    def _scrape_gauge(self, node: Node, metric: str) -> Optional[int]:
        url = f"http://{node.metrics_addr}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                text = resp.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            return None
        m = re.search(rf"^{metric}\{{[^}}]*\}}\s+(-?\d+)", text, re.MULTILINE)
        return int(m.group(1)) if m else None

    def scrape_epoch(self, node: Node) -> Optional[int]:
        return self._scrape_gauge(node, "lethe_cluster_epoch")

    def scrape_under_target(self, node: Node) -> Optional[int]:
        # W11.1: now a LIVE deficit (round.size - cursor), zeroed on
        # reconvergence — usable as a cheap convergence signal.
        return self._scrape_gauge(node, "lethe_replicas_under_target")

    def sample_probe(self, corpus: dict[bytes, bytes], alive: list[Node],
                     k: int = 50) -> ReplicaState:
        """probe_replicas over an evenly-spaced k-sample of the corpus —
        cheap enough to poll convergence at large N without N*nodes fetches."""
        items = list(corpus.items())
        step = max(1, len(items) // k)
        return self.probe_replicas(dict(items[::step]), alive)


# --------------------------------------------------------------------------
# Result plumbing.
# --------------------------------------------------------------------------


@dataclass
class Check:
    inv: str
    ok: bool
    detail: str
    warn: bool = False


@dataclass
class ScenarioResult:
    scenario: str
    checks: list[Check] = field(default_factory=list)

    def add(self, inv: str, ok: bool, detail: str, warn: bool = False) -> None:
        self.checks.append(Check(inv, ok, detail, warn))
        tag = "PASS" if ok else "FAIL"
        if ok and warn:
            tag = "WARN"
        print(f"  [{self.scenario}] {inv}: {tag} — {detail}", flush=True)

    @property
    def failed(self) -> list[str]:
        return sorted({c.inv for c in self.checks if not c.ok})

    @property
    def ok(self) -> bool:
        return not self.failed


# --------------------------------------------------------------------------
# Shared helpers.
# --------------------------------------------------------------------------


def _wait_node_ready(probe: ClusterProbe, node: Node, timeout_s: float = 30) -> bool:
    """Block until a node answers a trivial lookup (gRPC bound)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            probe.single(node).lookup([BlockId(hash=b"\x00" * 32)], request_id="ping")
            return True
        except Exception:  # noqa: BLE001 — readiness poll, any error => retry
            pass
        time.sleep(0.25)
    return False


def _restore_cluster(probe: ClusterProbe) -> None:
    """Best-effort: revive any stopped node, heal partitions, clear netem.
    Run in every scenario's finally so the next scenario starts clean."""
    for n in TOPOLOGY:
        try:
            if not is_running(n.container):
                revive(n.container, "sigkill")  # docker start
        except Exception:  # noqa: BLE001
            pass
        try:
            netem_clear(n.container)
        except Exception:  # noqa: BLE001 — no qdisc present is fine
            pass
    for i in range(len(TOPOLOGY)):
        for j in range(i + 1, len(TOPOLOGY)):
            try:
                partition_heal(TOPOLOGY[i].container, TOPOLOGY[j].container)
            except Exception:  # noqa: BLE001 — rule absent is fine
                pass


def _settle_replication(
    probe: ClusterProbe,
    corpus: dict[bytes, bytes],
    alive: list[Node],
    target: int,
    timeout_s: float = 8,
) -> None:
    """Wait until the corpus reaches `target` replicas across `alive` (best
    effort, used to establish a clean R=2 baseline before injecting faults)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        st = probe.probe_replicas(corpus, alive)
        if st.min_holders >= target and not st.corrupt and not st.lost:
            return
        time.sleep(0.3)


# --------------------------------------------------------------------------
# Scenario: node death (sigkill). Optionally restart and verify rejoin.
# This exercises every invariant.
# --------------------------------------------------------------------------


def scenario_kill(
    probe: ClusterProbe, *, mode: str = "sigkill", restart_after: float = 0.0
) -> ScenarioResult:
    name = "restart" if restart_after > 0 else "sigkill"
    res = ScenarioResult(name)
    victim = BY_ID["node2"]
    survivors = [n for n in TOPOLOGY if n is not victim]
    coordinator = survivors[0]
    corpus = make_corpus(f"w11-{name}")

    # Baseline: load + reach R=2 across the full cluster.
    probe.insert_corpus(corpus)
    _settle_replication(probe, corpus, TOPOLOGY, REPLICATION_FACTOR)
    base_epoch = probe.scrape_epoch(coordinator)
    base_hr = probe.ring_hit_rate(corpus)
    print(f"  [{name}] baseline: epoch={base_epoch} hit_rate={base_hr:.2f}", flush=True)

    # Inject: kill the victim. t0 is the instant of the kill.
    t0 = time.monotonic()
    kill(victim.container, mode)

    # Observe the recovery window.
    target = min(REPLICATION_FACTOR, len(survivors))  # R=2 across 2 survivors
    detect_ms: Optional[float] = None
    recovery_ms: Optional[float] = None
    min_hit_rate = base_hr
    data_loss_at: Optional[float] = None
    corruption_seen: dict[bytes, list[str]] = {}
    last_residual = len(corpus)  # blocks below the replication target
    samples = 0

    budget_ms = recovery_budget_ms(len(corpus))
    deadline = t0 + (budget_ms / 1000.0) + 2.0
    while time.monotonic() < deadline:
        now_ms = (time.monotonic() - t0) * 1000.0

        if detect_ms is None:
            ep = probe.scrape_epoch(coordinator)
            if ep is not None and base_epoch is not None and ep > base_epoch:
                detect_ms = now_ms

        st = probe.probe_replicas(corpus, survivors)
        last_residual = sum(1 for hs in st.holders.values() if len(hs) < target)
        if st.lost and data_loss_at is None:
            data_loss_at = now_ms
        if st.corrupt:
            corruption_seen.update(st.corrupt)
        if (
            recovery_ms is None
            and st.min_holders >= target
            and not st.lost
            and not st.corrupt
        ):
            recovery_ms = now_ms

        hr = probe.ring_hit_rate(corpus)
        min_hit_rate = min(min_hit_rate, hr)
        samples += 1

        if detect_ms is not None and recovery_ms is not None and samples >= 4:
            break
        time.sleep(0.2)

    # INV-1: never lost a block (data on zero alive nodes).
    res.add(
        "INV-1",
        data_loss_at is None,
        "no data loss across survivors"
        if data_loss_at is None
        else f"BLOCK LOST at t+{data_loss_at:.0f}ms (zero replicas)",
    )

    # INV-6: never served wrong bytes.
    res.add(
        "INV-6",
        not corruption_seen,
        "all served bytes match BlockId"
        if not corruption_seen
        else f"CORRUPTION on {len(corruption_seen)} block(s)",
    )

    # INV-2: detection (epoch bump) within budget.
    if detect_ms is None:
        res.add(
            "INV-2",
            False,
            f"epoch never advanced past {base_epoch} within {DETECTION_BUDGET_MS}ms",
        )
    else:
        res.add(
            "INV-2",
            detect_ms <= DETECTION_BUDGET_MS,
            f"epoch advanced at t+{detect_ms:.0f}ms (budget {DETECTION_BUDGET_MS}ms)",
        )

    # INV-3: re-replication converged to R=2 across survivors within the
    # size-aware honest budget (W11.1). A miss is a real regression now (the
    # budget is measured, not cosmetic), so it FAILS and reports the residual.
    if recovery_ms is None:
        res.add(
            "INV-3",
            False,
            f"did NOT reconverge to R={target} within {budget_ms:.0f}ms — "
            f"{last_residual}/{len(corpus)} block(s) still at R<{target} "
            "(non-convergence)",
        )
    else:
        res.add(
            "INV-3",
            recovery_ms <= budget_ms,
            f"reconverged to R={target} at t+{recovery_ms:.0f}ms "
            f"(budget {budget_ms:.0f}ms for {len(corpus)} blocks)",
        )

    # INV-5: load path never zeroed.
    res.add(
        "INV-5",
        min_hit_rate > 0.0,
        f"min ring hit-rate {min_hit_rate:.2f} over {samples} samples",
    )

    # INV-4: after detection settles, no survivor routes to the dead node.
    settle_target = t0 + INV4_SETTLE_MS / 1000.0
    while time.monotonic() < settle_target:
        time.sleep(0.1)
    stale = sum(
        probe.stale_remotehit_count(corpus, s, victim.node_id) for s in survivors
    )
    res.add(
        "INV-4",
        stale == 0,
        "no survivor returns RemoteHit->dead node"
        if stale == 0
        else f"{stale} stale RemoteHit(s) -> {victim.node_id} after {INV4_SETTLE_MS}ms",
    )

    # Optional restart: verify the node rejoins and the cluster reconverges.
    if restart_after > 0:
        wait = restart_after - (time.monotonic() - t0)
        if wait > 0:
            time.sleep(wait)
        revive(victim.container, mode)
        _wait_node_ready(probe, victim, timeout_s=30)
        rejoin_ok = False
        rj0 = time.monotonic()
        while time.monotonic() - rj0 < 12:
            st = probe.probe_replicas(corpus, TOPOLOGY)
            ep_v = probe.scrape_epoch(victim)
            if (
                st.min_holders >= REPLICATION_FACTOR
                and not st.lost
                and not st.corrupt
                and ep_v is not None
            ):
                rejoin_ok = True
                break
            time.sleep(0.3)
        hr_after = probe.ring_hit_rate(corpus)
        res.add(
            "INV-3",
            rejoin_ok,
            (
                f"victim rejoined; R>={REPLICATION_FACTOR} cluster-wide, "
                f"hit_rate={hr_after:.2f}"
            )
            if rejoin_ok
            else "victim did NOT reconverge after restart",
        )

    return res


# --------------------------------------------------------------------------
# Scenario: long pause (simulated stop-the-world / GC). Paused > dead_after,
# so it's detected dead; on unpause it must rejoin cleanly with a stale epoch.
# --------------------------------------------------------------------------


def scenario_pause(probe: ClusterProbe) -> ScenarioResult:
    res = ScenarioResult("pause")
    victim = BY_ID["node2"]
    survivors = [n for n in TOPOLOGY if n is not victim]
    coordinator = survivors[0]
    corpus = make_corpus("w11-pause")

    probe.insert_corpus(corpus)
    _settle_replication(probe, corpus, TOPOLOGY, REPLICATION_FACTOR)
    base_epoch = probe.scrape_epoch(coordinator)

    t0 = time.monotonic()
    kill(victim.container, "pause")

    detect_ms: Optional[float] = None
    data_loss = False
    corruption = False
    while time.monotonic() - t0 < (DEAD_AFTER_MS + 1500) / 1000.0:
        ep = probe.scrape_epoch(coordinator)
        if (
            detect_ms is None
            and ep is not None
            and base_epoch is not None
            and ep > base_epoch
        ):
            detect_ms = (time.monotonic() - t0) * 1000.0
        st = probe.probe_replicas(corpus, survivors)
        data_loss = data_loss or bool(st.lost)
        corruption = corruption or bool(st.corrupt)
        if detect_ms is not None:
            break
        time.sleep(0.2)

    res.add(
        "INV-2",
        detect_ms is not None,
        f"paused node detected dead at t+{detect_ms:.0f}ms"
        if detect_ms
        else "paused node never detected dead",
    )
    res.add(
        "INV-1",
        not data_loss,
        "corpus retrievable from survivors while paused"
        if not data_loss
        else "BLOCK LOST while a node was paused",
    )

    # Unpause and verify clean rejoin.
    revive(victim.container, "pause")
    _wait_node_ready(probe, victim, timeout_s=20)
    rejoin_ok = False
    rj0 = time.monotonic()
    while time.monotonic() - rj0 < 12:
        st = probe.probe_replicas(corpus, TOPOLOGY)
        ep_v = probe.scrape_epoch(victim)
        if (
            st.min_holders >= REPLICATION_FACTOR
            and not st.lost
            and not st.corrupt
            and ep_v is not None
        ):
            rejoin_ok = True
            break
        corruption = corruption or bool(st.corrupt)
        time.sleep(0.3)

    res.add(
        "INV-3",
        rejoin_ok,
        "unpaused node rejoined and reconverged to R=2"
        if rejoin_ok
        else "unpaused node did NOT reconverge",
    )
    res.add(
        "INV-6",
        not corruption,
        "no corruption through pause/unpause"
        if not corruption
        else "CORRUPTION observed through pause cycle",
    )
    return res


# --------------------------------------------------------------------------
# Scenario: network partition (node1 <-X-> node2; node0 reaches both).
#
# Under the no-consensus design the two isolated sides each mark the other
# dead and drop it from their ring — DIVERGENT VIEWS ARE EXPECTED, not a bug.
# A bridged partition like this moves NO data (no inserts during it), so it
# cannot lose redundancy or corrupt: writes are content-addressed, so even
# divergent routing can never produce conflicting bytes for a BlockId. The
# honest, falsifiable assertions are therefore:
#   INV-1  no data loss (every block stays retrievable from the host),
#   INV-6  no corruption,
#   INV-3  membership RECOVERS — after heal both isolated sides detect the
#          peer's return (a further epoch advance = resurrection), and the
#          bridging node (node0) never churns its view.
# "Reconverge to R=2" is NOT the right post-condition here: nothing dropped
# below R=2 in the first place, so it would assert nothing.
# --------------------------------------------------------------------------


def scenario_partition(probe: ClusterProbe) -> ScenarioResult:
    res = ScenarioResult("partition")
    a, b = BY_ID["node1"], BY_ID["node2"]
    observer = BY_ID["node0"]  # reaches both sides throughout
    corpus = make_corpus("w11-partition")

    probe.insert_corpus(corpus)
    _settle_replication(probe, corpus, TOPOLOGY, REPLICATION_FACTOR)
    base_epochs = {n.node_id: probe.scrape_epoch(n) for n in TOPOLOGY}

    data_loss = False
    corruption = False
    # Peak epoch each isolated side reaches DURING the partition (captures the
    # death-detection bump); rejoin must push past this after heal.
    peak = {a.node_id: base_epochs[a.node_id], b.node_id: base_epochs[b.node_id]}
    obs_churned = False

    partition_inject(a.container, b.container)
    try:
        t0 = time.monotonic()
        while time.monotonic() - t0 < (DEAD_AFTER_MS + 1500) / 1000.0:
            st = probe.probe_replicas(corpus, TOPOLOGY)  # all host-reachable
            data_loss = data_loss or bool(st.lost)
            corruption = corruption or bool(st.corrupt)
            for nid in (a.node_id, b.node_id):
                ep = probe.scrape_epoch(BY_ID[nid])
                if ep is not None and peak[nid] is not None and ep > peak[nid]:
                    peak[nid] = ep
            ep0 = probe.scrape_epoch(observer)
            if (
                ep0 is not None
                and base_epochs[observer.node_id] is not None
                and ep0 != base_epochs[observer.node_id]
            ):
                obs_churned = True
            time.sleep(0.3)

        bumped = [
            nid
            for nid in (a.node_id, b.node_id)
            if peak[nid] is not None
            and base_epochs[nid] is not None
            and peak[nid] > base_epochs[nid]
        ]
        print(
            f"  [partition] isolated sides that detected a death: {bumped or 'none'}; "
            f"bridging node {observer.node_id} stable={not obs_churned} "
            "(divergent views EXPECTED under no-consensus)",
            flush=True,
        )
    finally:
        partition_heal(a.container, b.container)

    # Heal: each isolated side must re-discover the other (resurrection =>
    # a further epoch advance past its during-partition peak).
    rejoined = False
    h0 = time.monotonic()
    while time.monotonic() - h0 < 12:
        st = probe.probe_replicas(corpus, TOPOLOGY)
        data_loss = data_loss or bool(st.lost)
        corruption = corruption or bool(st.corrupt)
        ea = probe.scrape_epoch(a)
        eb = probe.scrape_epoch(b)
        if (
            ea is not None
            and eb is not None
            and peak[a.node_id] is not None
            and peak[b.node_id] is not None
            and ea > peak[a.node_id]
            and eb > peak[b.node_id]
        ):
            rejoined = True
            break
        time.sleep(0.3)

    res.add(
        "INV-1",
        not data_loss,
        "no data loss through partition + heal"
        if not data_loss
        else "BLOCK LOST during partition",
    )
    res.add(
        "INV-6",
        not corruption,
        "no corruption through partition + heal"
        if not corruption
        else "CORRUPTION during partition",
    )
    res.add(
        "INV-3",
        rejoined and not obs_churned,
        f"both sides detected rejoin after heal "
        f"(epochs {a.node_id}>{peak[a.node_id]}, {b.node_id}>{peak[b.node_id]}); "
        f"bridge {observer.node_id} stable"
        if rejoined and not obs_churned
        else (
            "bridging node churned its view (false partition)"
            if obs_churned
            else "isolated sides did NOT detect rejoin after heal"
        ),
    )
    return res


# --------------------------------------------------------------------------
# Scenario: packet loss on one node. 5% is TOLERATED — the node must NOT be
# falsely declared dead (15 missed heartbeats over dead_after; P(all lost) is
# negligible at 5%), and the load path must stay alive with no corruption.
# 30% (heavy, opt-in) MAY kill the node; there death is acceptable and we only
# assert no corruption.
# --------------------------------------------------------------------------


def scenario_packet_loss(
    probe: ClusterProbe, *, pct: float = 5.0, tolerate_death: bool = False
) -> ScenarioResult:
    name = "packet_loss_heavy" if tolerate_death else "packet_loss"
    res = ScenarioResult(name)
    victim = BY_ID["node2"]
    corpus = make_corpus(f"w11-loss-{int(pct)}")

    probe.insert_corpus(corpus)
    _settle_replication(probe, corpus, TOPOLOGY, REPLICATION_FACTOR)
    base_epochs = {n.node_id: probe.scrape_epoch(n) for n in TOPOLOGY}

    mech = netem_loss(victim.container, pct)
    print(
        f"  [{name}] injecting {pct:.0f}% loss on {victim.node_id} via {mech}",
        flush=True,
    )
    try:
        false_death = False
        corruption = False
        min_hr = 1.0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15:
            for n in TOPOLOGY:
                if n is victim:
                    continue
                ep = probe.scrape_epoch(n)
                if (
                    ep is not None
                    and base_epochs[n.node_id] is not None
                    and ep > base_epochs[n.node_id]
                ):
                    false_death = True
            st = probe.probe_replicas(corpus, [n for n in TOPOLOGY if n is not victim])
            corruption = corruption or bool(st.corrupt)
            min_hr = min(min_hr, probe.ring_hit_rate(corpus))
            time.sleep(0.5)
    finally:
        netem_clear(victim.container)

    if tolerate_death:
        res.add(
            "INV-6",
            not corruption,
            "no corruption under heavy loss"
            if not corruption
            else "CORRUPTION under heavy loss",
        )
        res.add(
            "INV-5",
            True,
            f"min hit-rate {min_hr:.2f} under {pct:.0f}% loss (death tolerated)",
            warn=(min_hr == 0.0),
        )
    else:
        res.add(
            "INV-2",
            not false_death,
            f"no false death under {pct:.0f}% loss"
            if not false_death
            else f"FALSE DEATH: a survivor bumped epoch under {pct:.0f}% loss",
        )
        res.add(
            "INV-5",
            min_hr > 0.0,
            f"min ring hit-rate {min_hr:.2f} under {pct:.0f}% loss",
        )
        res.add(
            "INV-6",
            not corruption,
            "no corruption under packet loss"
            if not corruption
            else "CORRUPTION under packet loss",
        )
    return res


# --------------------------------------------------------------------------
# Scenario: LARGE working set (> kBoundedScan=256). This is the W11.1
# Finding-B regression test: before the periodic-sweep fix, a single death
# left the blocks beyond the 256-per-pass cap at R=1 forever. It must fully
# reconverge to R=2 now. Recovery is polled cheaply via the now-live
# replicas_under_target gauge + a sampled probe, then confirmed with a full
# probe; the budget is size-aware (recovery scales with the working set).
# --------------------------------------------------------------------------


def scenario_large(probe: ClusterProbe) -> ScenarioResult:
    res = ScenarioResult("large")
    victim = BY_ID["node2"]
    survivors = [n for n in TOPOLOGY if n is not victim]
    n = LARGE_CORPUS_SIZE
    corpus = make_corpus("w11_1-large", n)

    probe.insert_corpus(corpus)
    # Settle to R=2 on a sample (full settle would be N*nodes fetches).
    t_settle = time.monotonic()
    while time.monotonic() - t_settle < 20:
        if probe.sample_probe(corpus, TOPOLOGY).min_holders >= REPLICATION_FACTOR:
            break
        time.sleep(0.5)
    base = probe.sample_probe(corpus, TOPOLOGY)
    print(
        f"  [large] {n} blocks (> {REPLICATION_SCAN_CAP} cap), "
        f"baseline sample_min={base.min_holders}",
        flush=True,
    )

    budget_ms = recovery_budget_ms(n)
    target = min(REPLICATION_FACTOR, len(survivors))
    t0 = time.monotonic()
    kill(victim.container, "sigkill")

    recovery_ms: Optional[float] = None
    corruption = False
    deadline = t0 + budget_ms / 1000.0 + 5.0
    while time.monotonic() < deadline:
        smp = probe.sample_probe(corpus, survivors)
        if smp.corrupt:
            corruption = True
        if smp.min_holders >= target:
            # Sample looks converged — confirm against the FULL corpus.
            full = probe.probe_replicas(corpus, survivors)
            if full.corrupt:
                corruption = True
            if full.min_holders >= target and not full.lost:
                recovery_ms = (time.monotonic() - t0) * 1000.0
                break
        time.sleep(0.5)

    full = probe.probe_replicas(corpus, survivors)
    residual = sum(1 for hs in full.holders.values() if len(hs) < target)

    # INV-1 / INV-6: safety holds throughout.
    res.add(
        "INV-1",
        not full.lost,
        f"no data loss ({n} blocks, all on >=1 survivor)"
        if not full.lost
        else f"{len(full.lost)} block(s) LOST",
    )
    res.add(
        "INV-6",
        not corruption,
        "no corruption" if not corruption else "CORRUPTION on large set",
    )
    # INV-3: the Finding-B claim — FULL reconvergence beyond the 256 cap.
    res.add(
        "INV-3",
        recovery_ms is not None and recovery_ms <= budget_ms and residual == 0,
        f"all {n} blocks back to R={target} at t+{recovery_ms:.0f}ms "
        f"(budget {budget_ms:.0f}ms)"
        if recovery_ms is not None and residual == 0
        else f"INCOMPLETE: {residual}/{n} still at R<{target} after "
        f"{budget_ms:.0f}ms (Finding B would fail here pre-W11.1)",
    )
    return res


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

SCENARIOS = {
    "sigkill": lambda p: scenario_kill(p, mode="sigkill"),
    "restart": lambda p: scenario_kill(p, mode="sigkill", restart_after=6.0),
    "pause": scenario_pause,
    "partition": scenario_partition,
    "packet_loss": scenario_packet_loss,
    "large": scenario_large,
    "packet_loss_heavy": lambda p: scenario_packet_loss(
        p, pct=30.0, tolerate_death=True
    ),
}


def run_scenario(name: str) -> ScenarioResult:
    probe = ClusterProbe(TOPOLOGY)
    try:
        for n in TOPOLOGY:
            if not _wait_node_ready(probe, n, timeout_s=30):
                r = ScenarioResult(name)
                r.add("SETUP", False, f"{n.node_id} not ready before scenario")
                return r
        return SCENARIOS[name](probe)
    finally:
        _restore_cluster(probe)
        probe.close()


def main() -> None:
    # The ring-free probe clients have no peer map, so a survivor's RemoteHit
    # to another live node logs a "source_node unknown to client" warning per
    # block. That's benign here (we read source_node directly off .hits), but
    # it floods the suite output — quiet it to ERROR.
    logging.getLogger("lethe_client.client").setLevel(logging.ERROR)

    ap = argparse.ArgumentParser(description="Lethe chaos invariant checker")
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS) + ["all"])
    args = ap.parse_args()

    if args.scenario == "all":
        # The heavy (death-tolerated) loss probe is opt-in, not part of "all".
        names = [n for n in SCENARIOS if n != "packet_loss_heavy"]
    else:
        names = [args.scenario]

    all_failed: list[str] = []
    for name in names:
        print(f"\n=== scenario: {name} ===", flush=True)
        res = run_scenario(name)
        if res.failed:
            all_failed.extend(f"{name}:{inv}" for inv in res.failed)
        status = "FAIL" if res.failed else "PASS"
        print(
            f"RESULT scenario={name} status={status} "
            f"failed={','.join(res.failed) or 'none'}",
            flush=True,
        )

    print(f"\nSUITE failed={','.join(all_failed) or 'none'}", flush=True)
    sys.exit(1 if all_failed else 0)


if __name__ == "__main__":
    main()
