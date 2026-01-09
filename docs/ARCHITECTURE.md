# Architecture

```
                              Clients (vLLM workers)
                                       │
                              ┌────────┴────────┐
                              │  LetheClient    │
                              │  (Python, gRPC) │
                              │  + HashRing     │
                              └────────┬────────┘
                                       │ gRPC (Lookup/Insert)
                              ┌────────┴────────┐
                              │   gRPC Service  │
                              │      shim       │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   LetheCache    │  ←─── facade
                              └────────┬────────┘
                                       │
            ┌──────────┬──────────┬────┴─────┬──────────┬──────────┐
            │          │          │          │          │          │
       ┌────▼────┐ ┌───▼───┐ ┌────▼────┐ ┌───▼───┐ ┌────▼────┐ ┌──▼───┐
       │ Router  │ │ Replr │ │ Evictor │ │ Membr │ │ Tiered  │ │ RDMA │
       │ (cons   │ │ R=2   │ │ SIEVE   │ │ ship  │ │  Store  │ │ xprt │
       │  hash)  │ │       │ │ gossip  │ │ gossip│ │ HBM/DRAM│ │      │
       │         │ │       │ │         │ │       │ │  /SSD   │ │      │
       └─────────┘ └───────┘ └─────────┘ └───────┘ └─────────┘ └──────┘
                                                       │
                                                  ┌────┴────┐
                                                  │ BlockSt │ x3 (per tier)
                                                  └─────────┘
```

## Subsystem dependency layering

Subsystems are not symmetric peers — they form an acyclic dependency
layering. Each owns its own mutex and never reaches into another's lock;
a higher layer calls *down* into the API of a lower layer, never sideways
or up.

```
        Membership ──────────────► Router
            │                         │
            ▼                         ▼
        Replicator ───────────► TieredStore ───► BlockStore (×N tiers)
                                   ▲
        Evictor ───────────────────┘
        (also reads Membership for peer addresses)

        KvTransport       ← standalone; instantiated in main as
                            GrpcStreamTransport or IbverbsTransport,
                            handed to Replicator via a callback registry
        Metrics           ← leaf; everyone records, no one reads
```

This is what is meant by "no subsystem reaches into another's mutex": a
call from Membership into Router goes through Router's public API
(`SetPeers`), which acquires Router's own lock. The arrows above are
allowed *call* directions; reverse calls are bugs.

## Module responsibilities

- **LetheCache** (`cache.hpp`): facade owning all subsystems. Public RPC
  handlers live here; they delegate to the right subsystem.

- **Router** (`routing.hpp`): consistent hash ring with V=128 virtual nodes
  per peer. Maps `BlockId → (primary, replicas[])`. Rebuilds on
  membership change.

- **TieredStore** (`tiered_store.hpp`): composes three `BlockStore`s
  (HBM/DRAM/SSD). Handles tier selection on Put, promotion on Get,
  demotion before eviction.

- **BlockStore** (`block_store.hpp`): single-tier hash map with capacity
  accounting. Thread-safe.

- **Replicator** (`replication.hpp`): pushes blocks to replica successors;
  fetches from replicas on read-repair; triggers re-replication on
  membership change.

- **Evictor** (`eviction.hpp`): per-tier SIEVE eviction; gossips eviction
  events to peers.

- **Membership** (`membership.hpp`): heartbeat-based failure detection;
  bumps cluster epoch; calls into Router and Replicator on change.

- **KvTransport** (`kv_transport.hpp`): abstract bulk KV transport.
  `IbverbsTransport` (ibverbs, gated on `LETHE_ENABLE_RDMA=ON`) and
  `GrpcStreamTransport` (always built) are the concrete implementations.

- **Metrics** (`metrics.hpp`): prometheus-cpp pull metrics on
  `lethe_*` namespace; HTTP handler on metrics_port.

## Concurrency model

- One gRPC threadpool sized to `2 * num_cores`.
- Eviction loop: dedicated thread per tier (3 threads).
- Membership heartbeat: dedicated thread.
- RDMA completion polling: dedicated thread per peer.
- All shared state guarded by per-subsystem `shared_mutex` (reader-favored
  for hot reads on the routing ring and block store map).

## Data flow: a single Lookup

1. Client computes prefix-chained block IDs for the request's tokens.
2. Client `HashRing.route()` selects primary owner per block; for blocks
   sharing the same primary, batches into a single Lookup RPC.
3. Primary node `LetheCache::Lookup`:
   - Router checks `IsLocalPrimary`; for blocks where local node IS
     primary, looks up in TieredStore → LocalHit / Miss.
   - For blocks where local node is replica only (this only happens if
     the client's ring is stale, mid-reconfiguration), queries router for
     the *new* primary's address → RemoteHit.
   - On local Miss-but-I-am-primary, the Replicator is asked to
     `FetchFromAny(replica_set)` (read-repair, see below) before
     returning Miss to the client.
   - Returns response with hits + misses.
4. Client, on RemoteHit, opens StreamBlocks to source_node.
5. Client, on Miss, has vLLM recompute via prefill kernel.

## Sequence: read-repair on a primary-side miss

Read-repair happens when a Lookup arrives at the correct primary but the
primary has lost the block (eviction, restart, or never wrote it during
a transient outage). The primary attempts to recover from a replica
before reporting Miss; clients see one extra round-trip in the worst
case, not a recompute.

```
Client          Primary (LetheCache)        Replica_1            Replica_2
  │                     │                        │                    │
  │── Lookup(ids) ─────►│                        │                    │
  │                     │ Router.Route(id):      │                    │
  │                     │   primary = self       │                    │
  │                     │   replicas = [R1, R2]  │                    │
  │                     │                        │                    │
  │                     │ TieredStore.Get(id) → MISS                  │
  │                     │                        │                    │
  │                     │ Replicator.FetchFromAny(id, [R1, R2])       │
  │                     │── StreamBlocks ───────►│                    │
  │                     │── StreamBlocks ────────┼──────────────────►│
  │                     │                        │                    │
  │                     │◄── block bytes ────────│  (whichever ACKs first)
  │                     │ (other request cancelled)                   │
  │                     │                        │                    │
  │                     │ TieredStore.Put(block, hint=DRAM)           │
  │                     │ (local copy repaired; no rebroadcast — the  │
  │                     │  replicas already had it)                   │
  │                     │                        │                    │
  │◄── LocalHit ────────│                        │                    │
```

Failure cases:
- All replicas Miss → primary returns `Miss` to client; client recomputes
  via prefill.
- Primary fails mid-repair → client times out, retries (the ring's
  successor will not yet be promoted because no membership change fired;
  this is just one slow request).
- Replica returns a hash-mismatched block → treated as Miss for that
  replica; primary tries the next one. The corrupt copy is dropped on
  the replica via the same `OnEvictBroadcast` path (W11 chaos asserts
  this).
