# W4 perf regression diagnostic — 3× W1.4 wall-time

**Date:** 2026-05-27
**Status:** Closed — categorized as **(b) environmental**, no fix needed.

## The observation

During W4.1 verification (session of 2026-05-27), the W1.4 token-identical
regression PASSED but reported `wall=486s` against a stated "baseline" of
153s — roughly 3× slower. The W4.1 report flagged this as a perf concern
and explicitly deferred diagnosis to a pre-W5-6 micro-task. This doc is
that micro-task.

## What I did

Re-ran `pytest tests/correctness/test_token_identical.py` against current
HEAD (commits `fc42a76` through `edb6aee`, no source changes since the
W4.1 run) inside the same WSL2 / vLLM 0.19.1 / 4060 environment. No
instrumentation added — the existing `_run_vllm_for_w14.py` already records
`load_seconds` and `gen_seconds` per run, and the parent test records
`wall_seconds`. The breakdown is enough to localize where the time is
going without touching `LetheClient`.

## Numbers

| Run                              | Wall  | A load+gen      | B load+gen      | C load+gen      |
| -------------------------------- | ----- | --------------- | --------------- | --------------- |
| Original W1.4 baseline (stated)  | 153s  | (not preserved) | (not preserved) | (not preserved) |
| Pre-W4.1 dump on disk            | 170s  | 25.9 + 27.8     | 18.1 + 26.3     | 17.0 + 28.1     |
| W4.1 verification run (reported) | 486s  | (not preserved) | (not preserved) | (not preserved) |
| **This diagnostic rerun**        | 120s  | 26.0 + 14.1     | 14.7 + 13.7     | 13.0 + 13.7     |

The 486s figure did not reproduce. Current HEAD runs the same test in
120s — **faster than the original 153s baseline**, much faster than the
170s dump that pre-dated the W4.1 verification, and 4× faster than the
486s W4.1 number. The codepath is unchanged between runs; only the
environment differs.

## Where the time goes (current run)

- Gen time is essentially flat across A/B/C: 14.1s / 13.7s / 13.7s.
  Vanilla vLLM and connector-enabled vLLM spend the same wall on the
  generation loop. If `LetheClient.lookup()` were a hot spot, B and C
  would be visibly slower than A; they aren't.
- Load time is highest on Run A (26.0s) and lower on B/C (14.7s, 13.0s).
  This is the standard cold-disk-cache → warm-disk-cache pattern: A pays
  the model-shard fetch from disk; B/C reuse the page cache. Nothing
  here implicates Lethe.
- Test wall = 116s ≈ sum(per-run) 95s + ~21s subprocess spawn /
  lethe_server startup / pytest framing overhead. Accounted for.

Since gen times are A ≈ B ≈ C, the LetheClient hot path is not on the
critical path. **Per-primary batching produces one Lookup call per
generate() step in single-node mode (peers list is empty → ring is empty
→ all blocks fall into the single fallback-to-primary_address group);
that's the wire-cost floor and it's already there.** Instrumenting the
client would only confirm what the gen-time symmetry already shows.

## Categorization

**(b) Environmental.**

The 3× wall figure was a one-shot artifact, almost certainly some
combination of:

- WSL2 VM cold start / Hyper-V GPU partition handoff overhead.
- HuggingFace shard cache cold (the W4.1 run was the first model load
  of that WSL session; subsequent runs reuse the page cache).
- vLLM compile/warmup costs on a freshly-restarted CUDA context.
- Background WSL activity (Defender scan, `wsl --shutdown` reclaim, etc.)
  that the diagnostic environment can't isolate.

No code change between the 486s and 120s runs. No commits since W4.1
verification touch any path that W1.4 exercises (`fc42a76` adds a new
test file, `8b5b7f5` silences a clang -Werror in `replication.cpp`,
`edb6aee` is docs-only). The regression is not in Lethe.

## What this rules out

- **(a) Lethe-side overhead from W4 wiring** — ruled out. Gen-time
  symmetry across A/B/C in the rerun shows the connector path costs
  no measurable wall on the generation loop. If async replication or
  the new HashRing were eating time, B/C would diverge from A.
- **(c) vLLM-side or unrelated** — partially ruled out. vLLM's own
  per-call cost shows up the same in A as in B/C; nothing changed
  there. The 486s blip was specifically *this environment, that day*.

## What this leaves

Nothing actionable. The micro-task's decision tree was:
> If (b): we re-run to confirm and move on.

Confirmed. Moving on.

## Caveat for future-me

The W4.1 report's "3× slowdown" claim was based on a single observation
without a controlled re-run. In future verification gates, when a perf
number looks anomalous, **run it twice before reporting it as a
regression.** A 486s outlier with no code change and no environmental
control is noise, not signal. This costs one rerun (~2 min for W1.4)
and prevents a fake regression from steering subsequent work.

## Cross-references

- `tests/correctness/test_token_identical.py` (parent test)
- `tests/correctness/_run_vllm_for_w14.py` (child; already records
  `load_seconds` and `gen_seconds` per run)
- `tests/correctness/w1_4_results.json` (latest dump, this rerun)
- `docs/weekly/W3_W4.md` (W4 closeout that flagged this for follow-up)
