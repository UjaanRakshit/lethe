# Decisions

Non-obvious design decisions that future-me (or an interviewer who pushes
back) will want the why for. Append-only; date entries; don't rewrite
history. Per CLAUDE.md: "write each entry as if explaining it to a senior
infra engineer who will challenge it."

---

## 2026-05-26 — Heartbeat wire format: option (b), full PeerStatus mirror

**Context.** Session 0 audit found that `HeartbeatReply::alive_peers`
(C++) was originally `std::vector<std::string>` while
`HeartbeatResponse.alive_peers` (proto) was `repeated PeerStatus` with
three fields per entry (`node_id`, `last_seen_epoch`, `suspected`). The
gRPC shim would have to drop information one way and fabricate
defaults the other way.

**Decision.** Widen the C++ struct to mirror the proto field-for-field:
`std::vector<lethe::PeerStatus>` where `PeerStatus = {node_id,
last_seen_epoch, suspected}`. The same shape propagates to the Python
client (`lethe_client.epoch_watcher.PeerStatus` dataclass).

**Alternatives considered.**
- (a) **Downgrade the proto** to `repeated string alive_peers`. Cheaper
  in C++; loses the `suspected` bit on the wire, which the W11 chaos
  suite will want for assertions like "client marks the suspected peer
  down within N ms of the server suspecting it."
- (c) **Accept the lossy mapping at the shim boundary.** Cheapest now;
  pushes the problem to W11 when retrofit cost is highest. Considered
  and explicitly rejected.

**Rationale.** The wire is the contract. Widening the C++ struct is a
five-line change with no runtime cost; downgrading the proto throws
away signal we know the chaos suite needs; accepting lossy mapping is
"we'll fix it later" debt that compounds. The single bit of work that
this *does* introduce — `Membership::PeerInfo` had no `last_seen_epoch`
source field — was small and got fixed in the same session (commit C).

**Cross-references.** `proto/lethe.proto:128-144`,
`cache_server/include/lethe/types.hpp` (PeerStatus / HeartbeatReply),
`cache_server/src/main.cpp` (gRPC shim sketch),
`client/lethe_client/epoch_watcher.py` (Python mirror).

---

## 2026-05-26 — KvTransport: abstract base, two concrete derivatives

**Context.** The original `RdmaTransport` class was concrete and
ibverbs-only. CLAUDE.md hard rule 4 requires a gRPC-streaming fallback
for the data path so RDMA debugging cannot consume more than 2.5
weeks of W5-6. Naming the abstract base `RdmaTransport` while one
derivative is gRPC was confusing; naming the gRPC derivative
something like `RdmaTransport::Grpc` would lie about what it is.

**Decision.** The abstract base is `lethe::KvTransport` (in
`cache_server/include/lethe/kv_transport.hpp`). Concrete
implementations are:
- `GrpcStreamTransport` — always built; default when
  `LETHE_ENABLE_RDMA=OFF` or `RdmaIsAvailable()` returns false.
- `IbverbsTransport` — built only when `LETHE_ENABLE_RDMA=ON`.

The factory in `main.cpp` selects between them at startup.

**Alternatives considered.**
- Keep `RdmaTransport` as the abstract name. Rejected: gRPC isn't RDMA;
  the name was a lie that we'd carry forward forever.
- Use two unrelated classes with no shared base, dispatched via
  `std::variant`. Rejected: virtual dispatch is the natural pattern,
  and a transport interface is exactly the case it's designed for.

**Rationale.** "KvTransport" names what the abstract actually is — a
bulk-KV-block transport — without baking in the implementation. Cost
of the rename was one header move and ~20 lines of doc updates;
benefit is that no one reading `GrpcStreamTransport : public KvTransport`
will be confused.

**Cross-references.** `cache_server/include/lethe/kv_transport.hpp`,
`cache_server/src/grpc_stream_transport.cpp`,
`cache_server/src/ibverbs_transport.cpp` (renamed from
`rdma_transport.cpp` for the same reason).

---

## 2026-05-26 — PeerInfo holds two time concepts: wall-clock and cluster-epoch

**Context.** When the HeartbeatReply widening (above) introduced
`PeerStatus::last_seen_epoch` to the wire, `Membership::PeerInfo` had
to source that value. PeerInfo already had a `last_seen` field, but
that one was a `std::chrono::steady_clock::time_point` — wall-clock-ish
duration used by the suspect/dead detector. Cluster-epoch is a
monotone counter that bumps on membership changes. Different units;
different uses.

**Decision.** PeerInfo carries both:
- `std::chrono::steady_clock::time_point last_seen` — drives
  `suspect_after` and `dead_after` timer comparisons.
- `std::uint64_t last_seen_epoch` — what we ship to other nodes (and
  the Python client) so they can reason about staleness of our peer
  view across membership changes.

**Alternatives considered.**
- Derive `last_seen_epoch` from `last_seen` at reply-construction
  time. Rejected: there is no clean derivation; epoch advances only
  on membership change, not on every wall-clock tick.
- Use only the epoch counter and drop the wall-clock timestamp.
  Rejected: the suspect/dead detector fundamentally needs wall-clock
  duration (`now() - last_seen > suspect_after_ms`); cluster epoch
  doesn't tick fast enough.
- Use only the wall-clock and skip the epoch on the wire. Rejected:
  this is the option-(c) lossy mapping we already rejected above.

**Rationale.** Two time concepts because we have two time-sensitive
behaviors. The local failure detector wants "how long since we heard
from this peer" (wall-clock); cluster-wide gossip wants "as of which
epoch was this view consistent" (epoch). Conflating them costs more
than carrying both.

**Cross-references.** `cache_server/include/lethe/membership.hpp`
(PeerInfo doc comment), `cache_server/src/membership.cpp` (OnHeartbeat
sets both), `tests/unit/test_membership.cpp` (asserts monotonic
last_seen_epoch across heartbeats).
