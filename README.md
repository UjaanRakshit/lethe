# Lethe

Distributed prefix-aware KV cache for disaggregated LLM serving.

A 3-node cache layer that sits between vLLM prefill and decode workers. KV
blocks are sharded across nodes by prefix-aware consistent hashing, replicated
R=2, and tiered across HBM/DRAM/SSD. The design follows the architectures
described in Mooncake (ACM TOS '25), DistServe (OSDI '24), LMCache, and Llumnix
(OSDI '24) — built from the primitives rather than forked, to understand them.

## Architecture

```
                  ┌─────────────────────────────────────────────┐
                  │           Lethe Cluster (N=3)               │
                  │   ┌────────┐   ┌────────┐   ┌────────┐      │
                  │   │ node0  │◄─►│ node1  │◄─►│ node2  │      │
                  │   │ HBM    │   │ HBM    │   │ HBM    │      │
                  │   │ DRAM   │   │ DRAM   │   │ DRAM   │      │
                  │   │ SSD    │   │ SSD    │   │ SSD    │      │
                  │   └───▲────┘   └───▲────┘   └───▲────┘      │
                  └───────┼────────────┼────────────┼───────────┘
                          │  gRPC / RDMA (ibverbs)  │
                  ┌───────┴───────┐         ┌───────┴───────┐
                  │ vLLM Prefill  │────────►│ vLLM Decode   │
                  │   workers     │ KV xfer │   workers     │
                  └───────────────┘         └───────────────┘
```

## What it does

- **Prefix-aware routing.** Block IDs are `BLAKE3(prev_block_hash || tokens)`,
  so requests sharing a prefix produce identical block IDs and route to the
  same owner under consistent hashing (128 virtual nodes per peer) — no
  separate radix tree needed.
- **R=2 replication + read-repair.** Every block lives on a primary and one
  successor; reads repair missing replicas.
- **Tiered storage.** HBM → DRAM → SSD, with promotion on access count and
  demotion under pressure, plus cluster-wide SIEVE eviction.
- **Heartbeat membership.** 200 ms heartbeat, 3 s dead-detection, automatic
  re-replication of under-replicated blocks on failure. No consensus protocol.
- **vLLM integration.** A KV-transfer connector plugs Lethe in as an external
  prefix-cache tier for vLLM 0.19.1.
- **Transports.** gRPC by default; an ibverbs/RDMA data path over InfiniBand
  sits behind the `KvTransport` abstraction (`-DLETHE_ENABLE_RDMA=ON`).
- **Observability + chaos.** Prometheus/Grafana metrics and a fault-injection
  suite (kill, restart, pause, partition, packet loss).

## Results (measured)

Measured on a single NVIDIA L40S, `gemma-3-1b-it`, vLLM 0.19.1, median of 3
runs. The baseline is vLLM with its **native prefix cache ON** (not disabled),
working set swept past a single node's KV budget.

**Capacity crossover** — prefix-cache hit rate vs. working-set size:

| working set | 1 node (native cache) | Lethe (3 nodes, R=2) |
| ----------- | --------------------- | -------------------- |
| 1× node KV budget | 98.8% | 98.8% |
| 2× budget   | **0.0%** (collapses) | **98.8%** (sustains) |
| 4× budget   | **0.0%** | **85.2%** |

Lethe holds 2–4× the working set at a sustained hit rate, serving tens of
thousands of KV blocks from the distributed tier.

**Honest caveat:** at 1B-model scale this capacity win does **not** translate
to lower TTFT — recomputing a short prefill on a 1B model (~23 ms) is cheaper
than a loopback KV fetch (~53–81 ms). The latency benefit of a KV cache appears
only when per-token recompute cost exceeds fetch cost (larger model and/or
RDMA). The measurement is on 1B, and is reported as such.

**Failure recovery** — node kill → full R=2 reconvergence (loopback, median of
3): 3.7 s @ 200 blocks → 12.0 s @ 2000 blocks; ~3 s detection floor
(`dead_after`) plus a re-replication drain that scales with working set. (On
the docker bridge the fixed per-RPC latency dominates: 13–22 s, roughly flat.)

## Build

```bash
# Cache server (C++20). Default build needs only gRPC + protobuf (BLAKE3 is
# vendored). libibverbs is pulled in only with -DLETHE_ENABLE_RDMA=ON;
# Prometheus exposition is hand-rolled, so prometheus-cpp is not a dependency.
cmake -B build -S .
cmake --build build -j

# Client (Python 3.11+)
cd client && pip install -e .
```

## Run a 3-node local cluster

```bash
./scripts/run_cluster.sh        # docker-compose: 3 nodes + Prometheus + Grafana
```

## Failover demo (hit rate survives a node kill)

One command brings up the cluster, inserts a corpus at R=2, hard-kills a node,
and drives routed lookups while the cluster detects the death and
re-replicates. It reuses the chaos harness, so it runs the same code the suite
asserts against.

```bash
bash scripts/failover_demo.sh           # default scenario: sigkill
# then open Grafana at http://localhost:3000  (dashboard: "Lethe — Distributed KV Cache")
```

Watch, in order of clarity:

- **Cluster epoch** — steps up ~3.3 s after the kill (heartbeat detection,
  `dead_after=3 s`). The cleanest visual.
- **Under-replicated blocks** — spikes to the victim's in-route block count,
  then drains to 0 as the survivors restore R=2.
- **Cache hit rate** — stays high through the kill: survivors still hold the
  R=2 replicas, so the server-side hit ratio does not crash. (The scenario's
  stdout also prints its own client-side ring hit-rate, which dips ~1.00 →
  ~0.75 during the ~3 s detection window and recovers.)

Verified run (pure cache, no GPU): epoch bumped at t+3.5 s; 48 blocks
reconverged to R=2 by t+7.7 s; min ring hit-rate 0.75; all invariants pass.
For a longer continuous signal, `bash chaos/run_suite.sh` runs six scenarios
(~5 min) and leaves the stack up.

## Structure

```
lethe/
├── proto/          gRPC service definitions
├── cache_server/   C++20 cache server
├── client/         Python client + vLLM connector
├── disagg/         disaggregated prefill/decode orchestrator
├── benchmarks/     crossover + recovery harnesses
├── chaos/          fault-injection suite
├── tests/          correctness, integration, unit
├── deploy/         docker-compose, Prometheus, Grafana
└── scripts/        build, cluster, and RDMA setup helpers
```

## Out of scope

CXL, multi-tenancy with isolation, dynamic shard rebalancing under load, full
Raft for membership (static config + heartbeats is enough — production systems
delegate this to ZooKeeper/etcd), K8s deployment, autoscaling, custom
paged-attention kernels.

## References

- Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
  (KVCache.ai, ACM TOS 2025)
- DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM
  Serving (OSDI '24)
- Llumnix: Dynamic Scheduling for Large Language Model Serving (OSDI '24)
- LMCache: An Efficient KV Cache Layer for Enterprise-scale LLM Inference
- SIEVE is Simpler than LRU (NSDI '24)
- vLLM: Efficient Memory Management for LLM Serving with PagedAttention
  (SOSP '23)
```
