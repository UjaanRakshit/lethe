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

## Build

```bash
# Cache server (C++20, requires gRPC, protobuf, libibverbs, prometheus-cpp)
cmake -B build -S .
cmake --build build -j

# Client (Python 3.11+)
cd client && pip install -e .
```

## Run a 3-node local cluster

```bash
./scripts/run_cluster.sh        # docker-compose up; brings up 3 nodes + Prom + Grafana
python -m lethe_client.demo     # talks to the cluster
```

## Structure

```
lethe/
├── proto/              gRPC service definitions
├── cache_server/       C++20 cache server (the spine of W1–W8, W10)
├── client/             Python client + vLLM PagedAttention hook
├── disagg/             Disaggregated prefill/decode orchestrator (W9)
├── benchmarks/         ShareGPT/BurstGPT trace replay harness (W4, W12)
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
