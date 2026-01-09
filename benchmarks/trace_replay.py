"""ShareGPT trace replay harness (W4 baseline; W12 full sweep).

Replays the ShareGPT_V3 conversations dataset against vLLM with and without
Lethe enabled, comparing:

  - Cache hit rate
  - TTFT (time to first token), median + P99
  - Throughput (tokens/sec/node)
  - Tail latency under varying working-set sizes

The honest baseline (per the README): vLLM with its own native prefix cache
enabled vs. vLLM + Lethe, with the workload sized so that the working set
exceeds a single node's KV memory. Reporting "X× vs. vLLM with caching
disabled" is misleading; we don't do that.

Usage:
    python -m benchmarks.trace_replay \
        --trace workloads/sharegpt_v3.jsonl \
        --target lethe \
        --target-address localhost:50051 \
        --concurrency 16 \
        --max-requests 1000 \
        --out plots/results_lethe.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    cache_hit: bool


@dataclass
class Aggregate:
    target: str
    n_requests: int
    cache_hit_rate: float
    ttft_p50_ms: float
    ttft_p99_ms: float
    total_p50_ms: float
    total_p99_ms: float
    throughput_tok_s: float
    samples: list[Sample] = field(default_factory=list)


async def replay(args) -> Aggregate:
    samples: list[Sample] = []
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.monotonic()

    with open(args.trace) as f:
        lines = [json.loads(l) for l in f][: args.max_requests]

    async def run_one(req):
        async with sem:
            # TODO: dispatch via httpx to vLLM at args.target_address; record
            # TTFT from streaming response. For Lethe target, the vLLM
            # instance is configured with LetheCacheConnector and exposes a
            # `kv_hit` flag on the streamed metadata.
            return Sample(
                request_id=req["id"],
                prompt_tokens=len(req["prompt_tokens"]),
                output_tokens=req.get("output_tokens", 128),
                ttft_ms=0.0,
                total_ms=0.0,
                cache_hit=False,
            )

    samples = await asyncio.gather(*(run_one(r) for r in lines))
    duration = time.monotonic() - t0

    total_tokens = sum(s.output_tokens for s in samples)
    ttfts = [s.ttft_ms for s in samples]
    totals = [s.total_ms for s in samples]

    return Aggregate(
        target=args.target,
        n_requests=len(samples),
        cache_hit_rate=sum(1 for s in samples if s.cache_hit) / max(1, len(samples)),
        ttft_p50_ms=statistics.median(ttfts) if ttfts else 0,
        ttft_p99_ms=_p99(ttfts),
        total_p50_ms=statistics.median(totals) if totals else 0,
        total_p99_ms=_p99(totals),
        throughput_tok_s=total_tokens / max(1e-9, duration),
        samples=samples,
    )


def _p99(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[max(0, int(0.99 * len(s)) - 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--target", choices=["vllm-only", "lethe"], required=True)
    p.add_argument("--target-address", required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-requests", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    agg = asyncio.run(replay(args))
    args.out.write_text(json.dumps(asdict(agg), indent=2))
    print(f"hit_rate={agg.cache_hit_rate:.2%} "
          f"ttft_p50={agg.ttft_p50_ms:.1f}ms "
          f"ttft_p99={agg.ttft_p99_ms:.1f}ms "
          f"throughput={agg.throughput_tok_s:.0f} tok/s")


if __name__ == "__main__":
    main()
