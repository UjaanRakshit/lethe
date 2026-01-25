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

---

## 2026-05-27 — KvTransport adds Fetch alongside Send

**Context.** The W0 KvTransport interface had only `Send` (push) and an
`OnReceive` callback. The W4 Replicator's read-repair path (FetchFromAny)
did pull-shaped RPCs directly via its own gRPC PeerClient, bypassing the
abstraction. The W5-6 prompt mandated "Replicator takes KvTransport& at
construction. No more direct gRPC calls from Replicator."

**Decision.** Add `Fetch(peer_id, BlockId) -> future<optional<KvBlock>>`
to KvTransport, symmetric with `Send`. Both methods are implementable
over either gRPC or ibverbs without leaking transport specifics.
Replicator's FetchFromAny now dispatches `transport_->Fetch(...)` per
peer via `std::async`, preserving the W4 parallel-fan-out + first-non-
empty-wins semantic.

**Alternatives considered.**
- **Implement Fetch as Send-of-request + OnReceive-of-response.** Would
  shoehorn pull semantics through a push primitive. Rejected: it would
  force a request-correlation layer inside the transport (peer needs
  to match incoming response to its request future), which is wire-
  protocol territory and doesn't belong in the transport abstraction.
- **Keep FetchFromAny on its own gRPC PeerClient.** Rejected: violates
  the prompt's "no direct gRPC from Replicator" rule, and leaves the
  transport abstraction partially-applied (Send goes through it but
  Fetch doesn't) — which is exactly the kind of half-applied
  abstraction that decays.

**Rationale.** Push and pull are equally fundamental shapes for a bulk-
KV transport. The gRPC implementation makes Send → Insert RPC and
Fetch → Fetch RPC (the existing RPC surface already supports both).
The ibverbs implementation (W12) makes Send → ibv_post_send and Fetch
→ either a small Send-of-request-id followed by a Send-of-response, or
an RDMA Read against the peer's exported MR. Either ibverbs choice is
hidden behind the interface.

**Cross-references.** `cache_server/include/lethe/kv_transport.hpp`
(interface); `cache_server/src/grpc_stream_transport.cpp` (impl);
`cache_server/src/replication.cpp::FetchFromAny` (caller).

---

## 2026-05-27 — transport_ outlives replicator_ (cache.hpp field order)

**Context.** The W5-6 refactor moved peer-channel ownership from
`Replicator::peer_clients_` into `GrpcStreamTransport::Impl::peers`.
Replicator's worker threads call `transport_->Send` continuously
during the cache's lifetime. C++ destroys class members in REVERSE
declaration order; in the original `LetheCache` layout
(`...replicator_; ...transport_;`), `transport_` would be destroyed
FIRST, leaving Replicator's still-running workers with a dangling
pointer.

**Decision.** Reorder `cache.hpp` so `std::unique_ptr<KvTransport>
transport_;` is declared BEFORE `std::unique_ptr<Replicator>
replicator_;`. Reverse-declaration destruction then runs
`~Replicator` (joins workers) before `~KvTransport` (closes channels).
A load-bearing comment on the field block names the rule.

**Alternatives considered.**
- **Explicit `replicator_.reset()` in `LetheCache::Shutdown`.** Would
  destroy Replicator early enough. Rejected: introduces a window
  where `membership_` (which holds `Replicator*` for W8 use) has a
  dangling pointer. Even though Membership doesn't dereference it in
  W3-W4, the latent hazard is worse than the field reorder.
- **Reference-counted transport (shared_ptr).** Replicator's workers
  could hold their own shared_ptr to the transport so the transport
  lives until the last worker exits. Rejected: introduces ownership
  ambiguity into a layer that's supposed to be straightforward, just
  to work around an explicitly-orderable destruction.

**Rationale.** Member-order destruction is a C++ contract; encode the
invariant by declaration order and document it. The cost is one
comment block; the benefit is that the invariant survives future
refactors that move methods or rename things.

**Cross-references.** `cache_server/include/lethe/cache.hpp` (the
field block carries the load-bearing comment);
`cache_server/src/cache.cpp::Shutdown` (the destructor handler trusts
the order rather than re-establishing it).

---

## 2026-05-27 — W5-6 ships gRPC as the production data path

**Context.** The W5-6 milestone's primary deliverable was RDMA via
SoftRoCE. The dev environment is WSL2 on Windows; gate #1 of the
prompt required SoftRoCE be brought up within 90 minutes and produce
wire traffic via `ibv_rc_pingpong`. Otherwise: fall back to gRPC for
the data path.

**Decision.** Fall back. The WSL2 Microsoft kernel
(5.15.x-microsoft-standard-WSL2) ships without the InfiniBand
subsystem at all; `rdma_rxe` cannot be loaded; `rdma link show` fails
with a NETLINK_RDMA socket error. SoftRoCE is unreachable in this
environment.

The fallback still SHIPS the architectural piece: Replicator now
delegates ALL peer-to-peer block movement to KvTransport, and the
ibverbs class is wired into the build but its methods abort if
constructed. W12 PACE on real IB hardware swaps the transport with a
one-line change in `cache.cpp`.

**Alternatives considered.**
- **Build a custom WSL2 kernel with `CONFIG_RDMA_RXE=m`.** Plausible
  in principle (estimated 8-12 hours of kernel-build and Hyper-V-
  network-stack debugging, with significant probability of failing
  for unrelated reasons). Rejected: 25-hour W5-6 cap doesn't have
  room, and the bullet outcome is unchanged regardless.
- **Acquire IB hardware for this milestone.** Not in scope. W12 PACE
  is the venue for real-hardware validation.
- **Defer the KvTransport refactor and re-do it at W12.** Rejected:
  the abstraction was already half-built and the W4 Replicator was
  fully gRPC; finishing the abstraction now under no time pressure is
  cheaper than doing it at W12 under bring-up pressure. Also the
  bullet credibility — "designed and shipped the abstraction"
  outranks "designed an abstraction we never built."

**Rationale.** CLAUDE.md rule 4 is exactly this escape hatch ("if
SoftRoCE setup or QP debugging consumes more than 2.5 weeks of W5-6,
fall back to gRPC streaming for the data path"). The decision was
pre-authorized; gate #1 just removed the ambiguity about whether to
exercise it.

**Cross-references.** `docs/decisions/W5_rdma_fallback.md` (the full
gate-#1-firing record); CLAUDE.md "Hard rules" §4;
`cache_server/CMakeLists.txt` (the deferred-link comment naming the
exact CMake lines to re-add at W12).

---

## 2026-05-27 — LookupResult::Entry::local_data owns bytes (W7 change)

**Context.** W1 designed `local_data` as `std::span<const std::byte>`
borrowed into the store's `unordered_map<BlockId, KvBlock>` — valid as
long as no Erase or replacing Put happens on the same id. The lifetime
contract was that the gRPC shim serializes immediately, before any
concurrent mutation could land.

W7 adds the SSD tier (mmap'd slot allocator). SSD slots get reused on
Erase + Put — the bytes at a given mmap address can be overwritten by
an unrelated block once the slot's freed. Lending a span into that
storage is unsound past the next slot reuse.

**Decision.** Change `LookupResult::Entry::local_data` from
`std::span<const std::byte>` to `std::vector<std::byte>`. TieredStore's
`GetResult.data` is owned bytes; the per-tier `Get` paths copy into
this owned buffer. The W1 borrow contract is retired.

**Alternatives considered.**
- **Keep span at the upper layer; have SSD Get always promote to DRAM
  first** so the returned span lives in DRAM. Rejected: ties the
  promotion policy to the type of the return value (forced promotion
  on every SSD hit), conflicts with the access-count-threshold policy
  the prompt mandated.
- **Make GetResult a non-movable RAII handle that holds the bytes for
  HBM/DRAM via span and for SSD via owned buffer.** Rejected: the
  asymmetry leaks into every caller and makes lifetime reasoning
  per-tier. Uniform ownership is cheaper to reason about than
  conditional ownership.
- **Add a separate `local_data_owned` vector field only populated for
  SSD reads.** Rejected: every caller has to check both fields and
  know which one is live; the asymmetry never stops being a hazard.

**Rationale.** Per-Get memcpy of one block (≤ 64 KiB) is in the
noise. Long-context Lookups at hundreds of blocks per call are still
< 16 MiB of copy per RPC, well below per-RPC gRPC framing overhead.
The clarity of "Get returns owned bytes; lifetime is the GetResult"
beats every borrow-correctness invariant the span variant would have
needed.

**Cross-references.** `cache_server/include/lethe/cache.hpp:69-87`
(field declaration + W7 contract comment);
`cache_server/include/lethe/tiered_store.hpp` (GetResult);
`cache_server/src/cache.cpp::Lookup` (std::move into the entry);
`cache_server/src/main.cpp` Fetch handler (reads vector, no span).

---

## 2026-05-27 — SSD allocator: fixed 64 KiB slots + bump-then-freelist

**Context.** W7 needs an SSD-backed BlockStore with capacity bounded
by `cfg.ssd_bytes`. Three real choices: variable-size allocator with
a free-space data structure (B+-tree-style), a buddy allocator, or
fixed-size slots.

**Decision.** Fixed-size slots, default 64 KiB. Each slot is
`{SsdSlotHeader (64 bytes), payload (slot_bytes - 64)}`. Slot
allocation: bump pointer from index 0, plus a free list of indices
returned by Erase. Index is rebuilt at startup by scanning every slot
header and pulling live ones (magic byte == 0xA5).

**Alternatives considered.**
- **Variable-size allocator.** Real durable-storage systems do this
  (RocksDB, LevelDB). Rejected: ~3× the implementation complexity and
  drags in a free-space data structure with its own race conditions.
  W11's chaos suite is the venue for "does the SSD allocator fragment
  under realistic workload?"; W7 doesn't need the answer.
- **Buddy allocator.** Cleaner than variable-size but still more
  involved than a flat slot table. Same rejection rationale.
- **One file per block.** Rejected: filesystem metadata overhead per
  block, and N=million blocks = N inode entries.

**Rationale.** KV blocks at production block sizes (16-64 tokens
× model_dim × 2 (K+V) × bytes_per_param) range 8-128 KiB. A 64 KiB
slot fits the median case with room for header. Worst-case
fragmentation (50% under pessimistic block-size variance) is
acceptable for a 12-week project; W11 measures whether it actually
manifests. The freelist + bump-pointer combo is O(1) allocation
without a free-space search.

**Cross-references.** `cache_server/include/lethe/ssd_block_store.hpp`
(SsdSlotHeader + class declaration);
`cache_server/src/ssd_block_store.cpp::AllocSlot`;
`TieredStoreConfig::ssd_slot_bytes` (configurable per node, default
64 KiB).
