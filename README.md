# Lethe

Distributed prefix-aware KV-cache fabric for disaggregated LLM serving.

A 3-node cache layer sitting between vLLM prefill and decode workers. KV blocks
are sharded across nodes by prefix-aware consistent hashing; the design borrows
heavily from Mooncake (KVCache.ai, ACM TOS '25), DistServe (OSDI '24),
LMCache, and Llumnix (OSDI '24).

This is an execution-focused implementation of a current production
architecture, not a research project. The bullet is: production-shape
distributed infra from current papers.

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

## Week-by-week roadmap

| Week  | Deliverable                                                       |
| ----- | ----------------------------------------------------------------- |
| W1–2  | Single-node C++ cache, vLLM PagedAttention hook, token-identical  |
|       | correctness vs. vanilla vLLM                                      |
| W3–4  | 3-node distribution: consistent hashing on chunked prefix tokens, |
|       | R=2 replication, read-repair                                      |
| W5–6  | RDMA transport (SoftRoCE); benchmark the KV transfer path         |
| W7    | Tiered HBM/DRAM/SSD storage with tier-aware eviction              |
| W8    | Cluster-wide SIEVE eviction, gossip heartbeat, re-replication     |
| W9    | Disaggregated prefill/decode integration with two vLLM instances  |
| W10   | Observability: Prometheus, Grafana, structured tracing            |
| W11   | Chaos harness + failure-mode test suite                           |
| W12   | ShareGPT benchmark sweeps, design doc, failover demo, polish      |

Risk: W5–6 (RDMA). If SoftRoCE starts eating more than 2.5 weeks, fall back to
gRPC-over-TCP for the data path. The rest of the project stands.

## Results (measured)

Measured on PACE (Georgia Tech) ICE, one NVIDIA L40S per job, `gemma-3-1b-it`,
vLLM 0.19.1, median of 3 runs. Baseline is vLLM with its **native prefix cache
ON** (not disabled), working set swept past a single node's KV budget. Full
methodology and caveats in [docs/weekly/W12.md](docs/weekly/W12.md).

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
than a loopback KV fetch (~53–81 ms). The latency benefit of a KV cache
appears only when per-token recompute cost exceeds fetch cost (larger model
and/or RDMA), which was out of reach this window. We measured 1B and report 1B.

**Failure recovery** — node kill → full R=2 reconvergence (loopback, median
of 3): 3.7 s @ 200 blocks → 12.0 s @ 2000 blocks; ~3 s detection floor
(`dead_after`) + a re-replication drain that scales with working set. (On the
docker bridge the fixed per-RPC latency dominates: 13–22 s, roughly flat.)

## Build

```bash
# Cache server (C++20). Default build needs only gRPC + protobuf (BLAKE3 is
# vendored). libibverbs is pulled in only with -DLETHE_ENABLE_RDMA=ON;
# Prometheus exposition is hand-rolled, so prometheus-cpp is NOT a dependency.
cmake -B build -S .
cmake --build build -j

# Client (Python 3.11+)
cd client && pip install -e .
```

## Run a 3-node local cluster

```bash
./scripts/run_cluster.sh        # docker-compose up; brings up 3 nodes + Prom + Grafana
```

## Failover demo (hit rate survives a node kill)

One command brings up the cluster, inserts a corpus at R=2, hard-kills a node,
and drives routed lookups while the cluster detects the death and
re-replicates — W11's INV-5 made visual. It reuses the chaos harness, so it is
the same code the suite asserts against (no bespoke demo path).

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
  R=2 replicas, so the server-side hit ratio does not crash. That is INV-5.
  (The scenario's stdout also prints its own client-side ring hit-rate, which
  dips ~1.00 → ~0.75 during the ~3 s detection window and recovers; those
  client-side misses to the dead node never reach a live server, so they show
  in the log, not on the Grafana ratio panel.)

Verified run (pure cache, no GPU): epoch bumped at t+3.5 s; 48 blocks
reconverged to R=2 by t+7.7 s; min ring hit-rate 0.75; all six invariants PASS.
For a longer continuous signal, `bash chaos/run_suite.sh` runs six scenarios
(~5 min) and leaves the stack up.

## Structure

```
lethe/
├── proto/              gRPC service definitions
├── cache_server/       C++20 cache server (the spine of W1–W8, W10)
├── client/             Python client + vLLM PagedAttention hook
├── disagg/             Disaggregated prefill/decode orchestrator (W9)
├── benchmarks/         Crossover + recovery harnesses, plots (W12)
├── chaos/              Fault injection harness (W11)
├── tests/              Correctness, integration, unit
├── deploy/             docker-compose, Prometheus, Grafana (W10)
├── docs/               Design doc, architecture notes, benchmark results
└── scripts/            SoftRoCE setup, cluster startup, build helpers
```

## What's intentionally out of scope

CXL, multi-tenancy with isolation, dynamic shard rebalancing under load,
full Raft for membership (static config + heartbeats is enough — production
systems delegate this to ZooKeeper/etcd), K8s deployment, autoscaling,
custom paged-attention kernels.

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
