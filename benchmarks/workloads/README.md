# Benchmark workloads

> The capacity crossover claim runs on the **synthetic prefix workload**
> (below), via `benchmarks/crossover_sweep.py` — the cleaner instrument for
> that claim. ShareGPT and BurstGPT below are realistic-traffic follow-ups
> (download how-tos), not yet run.

## ShareGPT V3 (realistic-traffic follow-up — not yet run)

The conventional LLM-serving benchmark trace. ~90K real human-LLM
conversations with multi-turn structure, which exercises prefix caching well.
Run this to answer "does it hold on real traffic"; it is the wrong instrument
for a controlled capacity sweep (its uneven prefix-sharing confounds WSS).

Download:
```
huggingface-cli download --repo-type dataset \
    anon8231489123/ShareGPT_Vicuna_unfiltered \
    --local-dir ./sharegpt_raw
```

Preprocess (filters out non-English, deduplicates, tokenizes with the target
model's tokenizer, emits JSONL with one conversation per line):
```
python -m benchmarks.preprocess_sharegpt \
    --in sharegpt_raw \
    --out sharegpt_v3.jsonl \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

## BurstGPT (optional)

Captures burst patterns and inter-arrival distribution from real production
traces. Useful for showing that Lethe's failover and load balancing hold
under realistic spikes.

```
git clone https://github.com/HPMLL/BurstGPT
```

## Synthetic prefix workload (the capacity instrument)

`benchmarks/crossover_sweep.py` generates a controlled number of distinct
fixed-length prefixes and sweeps that count as a multiple of the single-node KV
budget. Because a prefix-cache hit depends on whether the working set fits —
not on token content — this isolates the capacity effect cleanly and is the
right tool for the crossover claim. It is *not* a realism claim: for "does it
hold on real traffic," run ShareGPT above. Don't present synthetic hit rates as
evidence about real-trace prefix distributions.
