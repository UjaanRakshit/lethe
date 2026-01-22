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

---

## 2026-05-27 — Async replication policy realized: fire-and-forget, bounded queue

**Context.** W0's "async by default" decision left the actual
mechanism unspecified. W4 had to land it. Three concrete questions
came up: (1) does Insert ACK before or after the replica push? (2)
what happens when replica peers are slow / overloaded? (3) what
happens when Insert load bursts past the replication thread pool's
capacity?

**Decision.**
- Insert returns immediately after the LOCAL tier write. Replica
  pushes happen on a background thread pool (N=4 workers).
- A bounded queue (depth 1024) sits between Insert and the workers.
  Push that would overflow is silently dropped (with a counter
  bump); W8's re-replication catches up after a peer's
  declared-dead transition.
- Per-Insert RPC deadline is 2s. Failed Inserts are dropped on
  the floor with a `replicate_failures` counter bump; CLAUDE.md rule
  forbids blocking scheduling on cache liveness.

**Alternatives considered.**
- **Sync replication** — block Insert ACK until R-1 replicas ACK.
  Rejected: changes the Insert latency profile from "single-node
  bytes write" to "cross-network ACK," which defeats the W0 design
  intent. Plus a slow replica becomes a load-bearing slow point.
- **Unbounded queue** — let memory grow. Rejected: a chaos-test
  partition would balloon memory on the surviving primary. The
  bounded queue gives back-pressure (dropping early) without ever
  blocking the producer.
- **Larger thread pool (N=16)** — context-switch overhead dominates
  on the tiny per-task work (one Insert RPC). N=4 mirrors the
  W1.4 connector executor sizing. Bump if W11 chaos shows worker
  starvation.

**Rationale.** Async + bounded + drop-on-overflow keeps Insert ACK
latency fast (the latency profile we sold in DESIGN.md) and bounds
worst-case memory regardless of replica health. The cost is that
dropped replications need W8's re-replication to repair under-
replicated blocks — but that's W8's job anyway, so we're not adding
new requirements, just relying on infrastructure W8 must build for
declared-dead recovery.

**Cross-references.** `cache_server/src/replication.cpp` (the pool +
ReplicateOut + queue), `cache_server/src/cache.cpp::Insert` (calls
ReplicateOut after the local write succeeds), `cache_server/include/
lethe/replication.hpp` (header doc updated to reflect "submitted
peers, not ACKed peers" return semantic).

---

## 2026-05-27 — gRPC connection pool: one channel per peer, lazy

**Context.** W4 needs to issue concurrent Insert (replication push)
and Fetch (read-repair pull) RPCs against peer nodes. Connection
pooling is a real lever: too few channels = head-of-line blocking;
too many = file-descriptor + socket overhead.

**Decision.** ONE `grpc::Channel` per peer, held by a `PeerClient`,
created lazily on first EnsurePeerClient call (which the
LetheCache ctor invokes for every static seed peer at startup).
The channel multiplexes RPCs internally over HTTP/2 streams; no
extra pool layer.

**Alternatives considered.**
- Channel pool (N channels per peer). Rejected: HTTP/2 already
  multiplexes thousands of concurrent streams per connection;
  parallel channels would only help if we hit HTTP/2's
  per-connection stream cap (default 100), which we won't at the
  W4 traffic level.
- Per-RPC channels. Rejected: TLS handshake / TCP setup cost would
  dominate every call.

**Rationale.** gRPC's design already does the pooling for us at
the stream level. The C++ Channel object handles auto-reconnect,
keepalive, and transient-failure backoff internally. We don't
out-think it.

**Cross-references.** `cache_server/src/replication.cpp::PeerClient`
(the channel + stub holder).

---

## 2026-05-27 — Client retry policy: 3 × 50 ms × 2^attempt, UNAVAILABLE-only

**Context.** Python LetheClient needs SOME retry policy for
transient gRPC failures or every long-context Lookup will see at
least one drop in production.

**Decision.** Three attempts max, 50 ms × 2^attempt exponential
backoff (50 ms, 100 ms — third attempt is the second retry).
Retry ONLY on UNAVAILABLE and DEADLINE_EXCEEDED. Every other code
(NOT_FOUND, FAILED_PRECONDITION, INVALID_ARGUMENT, INTERNAL, etc.)
surfaces as a per-block Miss / None response — never as a raised
exception into the connector's call stack.

**Alternatives considered.**
- Retry all codes. Rejected: an INVALID_ARGUMENT will never get
  better on retry; we'd just delay the inevitable. Worse, INTERNAL
  errors retried hot can mask data corruption.
- No retry. Rejected: TCP RST during a healthy operation
  (e.g. peer restart) would propagate as visible cache miss noise.
- Longer backoff (250 ms × 3). Rejected: vLLM scheduler can't
  afford a 750 ms tail on every cold-cache Lookup; 150 ms total
  budget for the retry tail is a more honest fit.

**Rationale.** UNAVAILABLE-and-DEADLINE_EXCEEDED-only matches gRPC's
own retry-eligibility taxonomy. The exponential backoff is the
standard pattern; the choice of 50 ms base is keyed to "single LAN
RTT + a buffer" rather than to any specific load model.

**Cross-references.** `client/lethe_client/client.py::_call_with_retry`.

---

## 2026-05-27 — Read-repair semantics on non-replica node: RemoteHit, never proxy

**Context.** A Lookup can arrive at a node that is NEITHER primary
NOR replica for the requested block — symptom of a client's HashRing
being stale during a membership change. The server has three choices:
(a) attempt to proxy the request to the actual primary; (b) return
RemoteHit pointing at the primary and let the client retry; (c)
return Miss.

**Decision.** Option (b). The server returns RemoteHit with
remote_node = primary, and the client refetches against the actual
primary. The server NEVER proxies cross-node Lookups.

**Alternatives considered.**
- Server-side proxy. Rejected: makes one Lookup into two-hop RPC
  with shared latency budget. Also makes any node potentially the
  hot spot for any block, which defeats the consistent-hash
  routing.
- Miss + force client retry against the right node. Rejected:
  the client doesn't know the right node without consulting the
  ring; a Miss would force a "where's primary" round-trip anyway.

**Rationale.** RemoteHit is the natural answer because (a) the
server already knows the primary via its own Router; (b) returning
it to the client preserves the proper consistent-hash data path;
(c) the client's transparent-fetch logic turns the two-RPC sequence
into one user-visible call with one extra round-trip.

**Cross-references.** `cache_server/src/cache.cpp::Lookup`
(distinguishes in-route vs out-of-route on miss);
`client/lethe_client/client.py::lookup` (transparent fetch on
RemoteHit).
