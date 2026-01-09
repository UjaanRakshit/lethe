# Benchmarks

## What we measure

| Metric | How | Why |
| --- | --- | --- |
| Cache hit rate | Lookups returning ≥1 hit, divided by total Lookups | The headline number |
| TTFT p50/p99 | Time from request submit to first generated token | What users feel |
| End-to-end p50/p99 | Submit to last token | Decode-bound regime |
| Throughput (tok/s/node) | Total output tokens / wall time / N_decode_workers | Capacity |
| Recovery time | Time from node death to all blocks back at R=2 | Failure story |
| Replication overhead | Insert RPC time with R=2 vs. R=1 | The cost of safety |

## Honest baselines

- **vanilla vLLM, no prefix cache** — only as a sanity check; do NOT report
  "X× faster than this" as the headline. vLLM has had prefix caching for
  over a year; this comparison is dishonest.
- **vLLM with native prefix cache, single node** — the real baseline.
  Workload sized so that the working set exceeds single-node KV memory,
  causing the native cache to thrash.
- **Lethe, 3 nodes, R=2** — the test condition.

The story: at WSS ≤ single-node memory, native prefix cache wins (no
network overhead). At WSS > single-node memory, Lethe wins because we
shard. Show the crossover empirically.

## Workloads

- ShareGPT V3 (primary; ~90K conversations, realistic prefix overlap).
- BurstGPT (secondary; tests behavior under traffic spikes).
- Synthetic prefix workload (smoke test only).

## What numbers I expect (no promises until measured)

- Hit rate at 2× single-node WSS: 45–65% Lethe vs. 15–25% native.
- TTFT improvement at that regime: 1.6–2.2×.
- Throughput at saturation: comparable per node; Lethe wins on aggregate.
- Recovery time, R=2, single-node kill, end-to-end:
  ~3.5s = ~3s detection (`dead_after`) + ~500ms re-replication.
  Earlier drafts of this file claimed "300–500ms"; that number was
  detection-only and ignored the heartbeat detector floor. The chaos
  suite is allowed to tighten `dead_after` for failover micro-benchmarks
  — clearly label those runs "aggressive detector" so the headline
  recovery number stays the conservative-defaults one.
- Routing overhead (Lookup p99): <5ms.

DO NOT chase Mooncake's published numbers (it claims up to 5x TTFT improvement
and 498% capacity increase in some configurations). Those were measured on
production hardware with their full optimization stack on real Moonshot traces.
We will not reach them in 12 weeks and should not pretend to.
