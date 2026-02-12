# Lethe — Design

## Why this exists

Disaggregated LLM serving (DistServe, Mooncake, Splitwise) separates the
compute-heavy prefill phase from the latency-critical decode phase. The KV
cache produced during prefill must be transferred to the decode worker;
the transfer cost — and the cost of *not* having the KV cache when a request
shares a prefix with previous traffic — dominates time-to-first-token for
long contexts. Lethe is the cache layer for this architecture.

We are not inventing anything here. The contribution is a clean, readable
implementation of primitives that exist in production at Moonshot
(Mooncake), AWS Bedrock, and Azure OpenAI but are not always cleanly
exposed in open-source form.

## Goals

1. Maintain high cache hit rate when the working set exceeds a single
   node's KV memory. W12 measured this (gemma-3-1b, L40S, median of 3):
   single-node native prefix caching collapses 98.8% → 0% once WSS crosses
   the node's KV budget, while Lethe sustains 98.8% at 2× and 85.2% at 4×.
   The win is capacity, not latency — at 1B scale recompute is cheaper than
   a network fetch, so TTFT does not improve; see docs/weekly/W12.md.
2. Survive single-node failure under R=2 with **sub-3s detection
   (`dead_after`) + full R=2 reconvergence that is bounded ~constant in
   working-set size**. W11.1 measured reconvergence at ~13–22s across
   200–2000 blocks on the docker bridge — roughly flat, because
   re-replication throughput scales with load (~15 blk/s at N=200 up to
   ~131 blk/s at N=2000). It does NOT degrade at scale. Earlier drafts
   claimed "<500ms" (detection-only) and then "≤3.5s end-to-end"
   (3s detect + 500ms re-replicate); BOTH are retired — the 500ms
   re-replication slice was only ever true at tiny N. See
   docs/weekly/W11_1.md for the measured curve and CLAUDE.md
   "Architecture spine." Re-replication completeness holds at any working
   set as of W11.1 (the periodic sweep covers sets past the per-pass cap).
3. Integrate with vLLM as a drop-in KV transfer connector. The
   correctness claim is *cache-equivalence*, not bit-equivalence: given
   the same set of cache hits vs. misses, Lethe + vLLM produces the same
   tokens as vLLM doing those same hits via its native prefix cache.
   Floating-point reductions in attention are non-associative on GPU;
   chasing across-cache-boundary bit-identicality is a losing trade for
   a 12-week project. The `tests/correctness/test_token_identical.py`
   fixture fixes the hit/miss schedule and asserts equivalence under
   that schedule.
4. Provide visibility (Prometheus + Grafana) and a chaos harness so the
   correctness story is defensible under interview pressure.

## Non-goals

- Replacing vLLM's scheduler. Lethe sits below the scheduler.
- Cross-tenant isolation. Single-tenant only.
- Geo-replication. Single-rack, single-DC.
- Dynamic shard rebalancing under load. Static peer set + re-replication
  on failure is enough.
- Consensus on membership. Heartbeat + static seed list is enough; in
  production you'd delegate this to etcd.

## Key design decisions

### 1. Prefix-chained block hashing

Each KV block's identifier is `H(prev_block_hash || tokens_in_this_block)`.
Two requests sharing the first k tokens produce identical block IDs for the
first `ceil(k/block_size)` blocks. Under consistent hashing on the block ID,
this gives prefix-aware routing for free without a dedicated trie or radix
structure.

Hash function: BLAKE3 (32-byte output). Fast, content-addressable, no
collision risk at our scale.

**What the router hashes.** A `BlockId` carries
`(content_hash, layer, head_group, model_id)`. The router hashes
`content_hash` and *only* `content_hash`. The other three fields are
in-memory disambiguators on the block-store map — they prevent KV bytes
for layer 0 and layer 31 from colliding under the same key — but they
do not affect routing. All layers of one prefix block therefore share a
primary owner. We picked this over per-layer routing so that the client
can batch every layer of one logical block into a single Lookup RPC; the
tradeoff is a "fat" primary per prefix, which the consistent-hash ring's
128 virtual nodes per peer is meant to smear across the cluster.

### 2. Consistent hashing with virtual nodes (V=128)

Standard ring: each peer maps to 128 virtual nodes on a 64-bit hash ring.
Block ID is hashed to a 64-bit key; primary owner is the next clockwise
virtual node; replicas are the next R-1 distinct peers.

Why 128 vnodes: gives <10% load imbalance for a 3-node cluster.

### 3. Replication factor R=2

Every block lives on its primary owner and one successor. R=3 would be
defensible but costs 50% more memory; for 3 nodes, R=2 already gives
single-failure tolerance and matches Mooncake's published config.

### 4. Cluster-wide SIEVE eviction

Per-node SIEVE is from NSDI '24 — single-pointer scan with a visited bit,
competitive with LRU at lower overhead. The cluster-wide piece: when we
evict, gossip the eviction so peers don't read-repair from us. This is
best-effort; the cost of a missed broadcast is one wasted RPC.

### 5. Heartbeat membership, no consensus

200ms heartbeat. 1s to suspected. 3s to declared dead. On declaration,
cluster epoch increments, router rebuilds, replicator triggers
re-replication of under-replicated blocks. No leader election. The peer set
is static at startup; adding nodes requires a restart (acceptable
for a demo / interview project).

### 6. gRPC for control plane, RDMA for KV transfer

Lookup, Insert, Heartbeat, EvictBroadcast — gRPC. They're small and
latency-tolerant.

StreamBlocks — RDMA via SoftRoCE. KV blocks are large (MBs each at long
context) and the transfer is throughput-critical. If RDMA setup blows the
W5–6 budget, fall back to gRPC streaming with the same interface.

### 7. Tiered HBM/DRAM/SSD storage

- HBM (GPU memory): hottest blocks. Tiny. Optional — disabled if no GPU.
- DRAM: working set. Most blocks live here.
- SSD: cold tail. Demoted from DRAM when hot.

Promotion policy: after `promotion_access_threshold` accesses (default 2)
in a tier, promote to next-faster tier on eviction pressure of that tier.

**HBM allocator.** Two build modes:

- **Default (`LETHE_ENABLE_CUDA=OFF`):** the HBM `BlockStore` is backed
  by pinned host memory. This lets the tier plumbing, promotion paths,
  and metrics be exercised without a GPU on the box, which keeps W7
  developable on the laptop. Benchmarks that use this mode MUST tag the
  result "HBM=pinned-host" so it cannot be passed off as device-memory
  performance.
- **Stretch (`LETHE_ENABLE_CUDA=ON`):** the HBM tier uses real device
  memory via `cudaMalloc`, and the W9 disagg integration can move blocks
  into vLLM's PagedAttention buffers without a host roundtrip. This is
  the goal but not the W7 requirement.

If `hbm_bytes == 0` the tier is disabled entirely (the `BlockStore`
pointer is null), regardless of build mode.

## Failure modes and what happens

| Failure | Detection | Recovery |
| --- | --- | --- |
| Single node crash | Heartbeat timeout (3s) | Router rebuilds; replicator restores full R=2 on alive peers via a periodic re-replication sweep (W11.1). Measured ~13–22s end-to-end, ~flat across 200–2000 blocks (docker bridge); completeness holds at any working-set size |
| Network partition (3-node) | Both sides suspect each other | No new replicas accepted on minority side; clients retry on healed |
| Disk full on SSD tier | Insert fails with rejected | Caller falls back to DRAM-only or recompute |
| Slow node (long GC) | Heartbeat suspected, then dead | Same as crash |
| Corrupted block | Hash mismatch on read | Treat as miss, request recompute |

## What's intentionally fragile

- Eviction broadcasts are gossiped, not Paxos'd. A lost broadcast costs one
  RPC.
- Cluster epoch is incremented locally on each membership change; it's
  monotonic per node but not globally synchronized. Read-repair tolerates
  this.
- We don't fsync SSD writes. A node crash loses recently-written SSD blocks.
  Acceptable because the source of truth (the LLM weights) can recompute.

## What we'd do differently in production

(For the interview: "this is the version that's appropriate for a 12-week
project. In production you would...")

- etcd/ZooKeeper for membership.
- A real shard-rebalancing controller (Kafka KIP-500 style).
- Cross-DC replication with conflict-free merging.
- Multi-tenancy with per-tenant quotas and prefix-cache isolation.
- A real B+tree on disaggregated memory (CXL or RDMA) for the SSD tier.
