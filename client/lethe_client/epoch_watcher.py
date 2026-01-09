"""Client-side cluster-epoch watcher and HashRing refresher.

The client mirrors the cache server's consistent-hash ring (see
`lethe_client.routing.HashRing`). When membership changes server-side
(node death, recovery), the server bumps its cluster epoch and rebuilds
its ring; the client must do the same or it will route to dead nodes.

This module runs a 1Hz background poll against any reachable peer's
Heartbeat RPC, learns the current epoch + alive peer set, and atomically
swaps the client's HashRing when the epoch changes. The poll interval
intentionally matches `MembershipConfig::heartbeat_interval` (200ms) ×5
so that worst-case staleness is one second — well inside the 3.5s
recovery budget (CLAUDE.md "Architecture spine").

W3 wires this in; W8 makes it failure-aware (skip unreachable peers,
back off on repeated failures). W11 chaos tests assert that the client's
ring converges within `dead_after + 1s` after a node kill.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .routing import HashRing

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerStatus:
    """Mirror of the C++ `lethe::PeerStatus` (cache_server/include/lethe/types.hpp)
    and proto `message PeerStatus` (proto/lethe.proto). Field-for-field.
    """

    node_id: str
    last_seen_epoch: int = 0
    suspected: bool = False


@dataclass
class EpochSnapshot:
    """One server's reported view of cluster state. Mirror of C++
    `HeartbeatReply` — `alive_peers` carries full PeerStatus, not bare
    node IDs, so the client can see who the server suspects without
    waiting for that node to be declared dead.
    """

    source_peer: str
    epoch: int
    alive_peers: list[PeerStatus] = field(default_factory=list)


# Type alias for the heartbeat probe — injected so tests can fake it
# without standing up real gRPC channels.
HeartbeatProbe = Callable[[str], Optional[EpochSnapshot]]


class EpochWatcher:
    """Background poller that keeps a HashRing in sync with the server.

    Lifecycle: construct → start() → (background thread polls) → stop().
    Thread-safe: `current_epoch` and `ring` are read under an internal
    lock; the ring is swapped wholesale on epoch change rather than
    mutated in place, so readers never see a half-built ring.
    """

    def __init__(
        self,
        seed_peers: list[str],
        probe: HeartbeatProbe,
        ring: HashRing,
        poll_interval_seconds: float = 1.0,
    ):
        self._seed_peers = list(seed_peers)
        self._probe = probe
        self._poll_interval = poll_interval_seconds

        self._lock = threading.Lock()
        self._ring = ring
        self._epoch = 0
        # Seed peers don't yet have a server-reported PeerStatus, so
        # synthesize one with sane defaults; the first successful probe
        # replaces this list wholesale.
        self._alive_peers: list[PeerStatus] = [
            PeerStatus(node_id=p) for p in seed_peers
        ]

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- Public API -------------------------------------------------------

    @property
    def current_epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def ring(self) -> HashRing:
        # Snapshot under the lock so concurrent set_peers() can't tear it.
        with self._lock:
            return self._ring

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="lethe-epoch-watcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
            self._thread = None

    # ---- Internals --------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            snap = self._probe_any_peer()
            if snap is not None:
                self._apply_if_newer(snap)
            # Sleep interruptible by stop() — never block past shutdown.
            self._stop_evt.wait(self._poll_interval)

    def _probe_any_peer(self) -> Optional[EpochSnapshot]:
        # Walk the currently-known alive (non-suspected) set first; fall
        # back to all known peers (including suspected); then to seeds.
        with self._lock:
            known = list(self._alive_peers)
            seeds = list(self._seed_peers)
        candidate_ids = [p.node_id for p in known if not p.suspected]
        if not candidate_ids:
            candidate_ids = [p.node_id for p in known]
        if not candidate_ids:
            candidate_ids = seeds
        for peer_id in candidate_ids:
            try:
                snap = self._probe(peer_id)
            except Exception as e:  # noqa: BLE001 — probe is foreign code
                logger.debug("epoch probe to %s failed: %s", peer_id, e)
                continue
            if snap is not None:
                return snap
        return None

    def _apply_if_newer(self, snap: EpochSnapshot) -> None:
        with self._lock:
            if snap.epoch <= self._epoch:
                return
            logger.info(
                "cluster epoch %d → %d (from %s, peers=%s)",
                self._epoch, snap.epoch, snap.source_peer,
                [p.node_id for p in snap.alive_peers],
            )
            self._epoch = snap.epoch
            self._alive_peers = list(snap.alive_peers)
            # The ring only routes by node_id; suspected peers stay in the
            # ring until they're declared dead server-side, otherwise a
            # transient probe failure would cause cluster-wide reshuffles.
            ring_peers = [p.node_id for p in self._alive_peers]
            # Swap, don't mutate — concurrent readers must see one or the
            # other ring, never a half-built one.
            self._ring = HashRing(
                peers=ring_peers,
                vnodes_per_peer=self._ring.vnodes_per_peer,
            )
