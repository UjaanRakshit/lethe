# Benchmark workloads

## ShareGPT V3 (primary)

The conventional LLM-serving benchmark trace. ~90K real human-LLM
conversations with multi-turn structure, which exercises prefix caching well.

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

## BurstGPT (optional, W12)

Captures burst patterns and inter-arrival distribution from real production
traces. Useful for showing that Lethe's failover and load balancing hold
under realistic spikes.

```
git clone https://github.com/HPMLL/BurstGPT
```

## Synthetic prefix workload (smoke test)

`benchmarks.synth_prefix` generates conversations with controllable prefix
overlap rate (0%, 25%, 50%, 90%) to characterize cache behavior cleanly. Use
this for sanity checks; never report only synthetic numbers.
