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

## Measured numbers (W12)

PACE ICE, one NVIDIA L40S, `gemma-3-1b-it`, vLLM 0.19.1, median of 3. Baseline
is vLLM native prefix cache ON. Full method + caveats: `docs/weekly/W12.md`.

**Capacity crossover** — prefix-cache hit rate vs. working-set size:

| WSS (× single-node KV budget) | 1 node, native cache | Lethe 3-node R=2 |
| ----------------------------- | -------------------- | ---------------- |
| 1×   | 98.8% | 98.8% |
| 2×   | 0.0% (collapses) | 98.8% (sustains) |
| 4×   | 0.0% | 85.2% |

The earlier estimates on this line ("45–65% Lethe vs 15–25% native") were
wrong in both directions: native collapses harder (to 0%, a clean cliff once
WSS crosses the budget) and Lethe sustains higher (85–99%). Hit was confirmed
real, not just present: config B served 17 320 KV blocks from Lethe at 2× and
30 091 at 4× (`meta_load_blocks`).

**TTFT — the capacity win does NOT convert to a latency win at 1B scale.**
Native is flat ~23 ms even at 0% hit (recompute of a 256-token prefill on a
1B model is cheap); Lethe is 53–81 ms (loopback fetch). The earlier "1.6–2.2×
TTFT improvement" estimate is **retired** — it does not hold at this scale.
The latency benefit requires per-token recompute cost to exceed fetch cost
(larger model and/or RDMA). We measured 1B; we report 1B. Reporting a TTFT
"win" here would require the dishonest cache-OFF baseline (rule 1) — we don't.

- Recovery time, R=2, single-node kill, end-to-end. Two environments:

  **W12 (PACE loopback, 3 processes, median of 3):** 3.7 s @ 200 blocks,
  4.4 s @ 500, 7.6 s @ 1000, 12.0 s @ 2000. Loopback strips the docker
  bridge's fixed per-RPC latency, so recovery is 3–4× faster *and* the
  per-block re-replication cost becomes visible (grows with N). ~3 s
  detection floor (`dead_after`) is consistent with the bridge curve.

  **W11.1 (docker bridge, fresh cluster, full R=2 reconvergence):**

  | working set | recovery | residual | rate |
  | ----------- | -------- | -------- | ---- |
  | 200 blocks  | 13.4s    | 0        | ~15 blk/s  |
  | 500 blocks  | 21.8s    | 0        | ~23 blk/s  |
  | 1000 blocks | 22.0s    | 0        | ~45 blk/s  |
  | 2000 blocks | 15.3s    | 0        | ~131 blk/s |

  = ~3s detection (`dead_after`) + a re-replication drain that is **roughly
  FLAT (~10–19s) across 200–2000 blocks** — throughput scales with load, so
  recovery does NOT grow with working-set size. Earlier drafts claimed
  "300–500ms" (detection-only) and then "~3.5s = 3s + 500ms re-replicate";
  BOTH retired — the 500ms slice was only true at tiny N. Loopback (the
  W7-W8 `test_failover_recovery` at N=200) is faster than this docker-bridge
  curve. The chaos suite may tighten `dead_after` for detection micro-
  benchmarks — label those "aggressive detector" so the headline number
  stays the conservative-defaults one. Deferred re-replication speedups
  (scope-to-dead-peer, suppress the ingest cascade, batch the push path) are
  the levers if this needs to come down — see docs/DECISIONS.md (W11.1).
- Routing overhead (Lookup p99): <5ms.

DO NOT chase Mooncake's published numbers (it claims up to 5x TTFT improvement
and 498% capacity increase in some configurations). Those were measured on
production hardware with their full optimization stack on real Moonshot traces.
We will not reach them in 12 weeks and should not pretend to.
