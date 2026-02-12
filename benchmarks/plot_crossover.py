"""Consolidate + plot the W12 capacity-crossover sweep.

Reads W12_sweep_<CONFIG>_n<NDIST>_<REP>.json (written by crossover_sweep.py),
takes the median over reps per (config, n_distinct), and writes:
  W12_crossover.json  the median + raw-run table behind the plot
  W12_crossover.png   two panels: hit rate vs WSS, and TTFT vs WSS

Hit-rate panel is the headline (native collapses past the single-node budget;
Lethe sustains). TTFT panel is the honest caveat (at 1B scale a loopback fetch
is slower than recompute, so the capacity win does not become a latency win).

Usage: python benchmarks/plot_crossover.py [SRC_DIR] [OUT_DIR]
(both default to benchmarks/plots/).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT = os.path.join(_REPO, "benchmarks", "plots")
SRC = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
OUT = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT
CONFIG_LABEL = {"A": "vLLM native prefix cache (1 node)",
                "B": "Lethe 3-node distributed cache"}


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(SRC, "W12_sweep_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        runs.setdefault((r["config"], int(r["n_distinct"])), []).append(r)

    consolidated = {}
    for (cfg, nd), rs in sorted(runs.items()):
        consolidated.setdefault(cfg, {})[nd] = {
            "n_reps": len(rs),
            "wss_tokens": rs[0].get("wss_tokens"),
            "hit_rate_median": med([r.get("hit_rate") for r in rs]),
            "hit_rate_runs": [r.get("hit_rate") for r in rs],
            "ttft_p50_median": med([r.get("ttft_p50_ms") for r in rs]),
            "ttft_p99_median": med([r.get("ttft_p99_ms") for r in rs]),
            "meta_load_blocks_median": med([r.get("meta_load_blocks") for r in rs]),
        }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "W12_crossover.json"), "w") as fh:
        json.dump(consolidated, fh, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for cfg in ("A", "B"):
        if cfg not in consolidated:
            continue
        nds = sorted(consolidated[cfg])
        ax1.plot(nds, [(consolidated[cfg][n]["hit_rate_median"] or 0) * 100 for n in nds],
                 marker="o", label=CONFIG_LABEL[cfg])
        ax2.plot(nds, [consolidated[cfg][n]["ttft_p50_median"] for n in nds],
                 marker="o", label=CONFIG_LABEL[cfg])
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("working set (distinct 256-token prefixes)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    ax1.set_ylabel("prefix-cache hit rate (%)")
    ax1.set_title("Hit rate vs working set (median of 3)")
    ax1.set_ylim(-5, 105)
    ax2.set_ylabel("TTFT p50 (ms)")
    ax2.set_title("TTFT vs working set (median of 3)")
    fig.suptitle("Lethe W12: distributed cache sustains hit rate past the single-node "
                 "KV budget\n(gemma-3-1b, L40S; TTFT caveat: loopback fetch > 1B recompute)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "W12_crossover.png"), dpi=130)

    for cfg in ("A", "B"):
        if cfg not in consolidated:
            continue
        print(f"\n=== {cfg}: {CONFIG_LABEL[cfg]} ===")
        for nd in sorted(consolidated[cfg]):
            c = consolidated[cfg][nd]
            print(f"  n={nd:5d}  wss={c['wss_tokens']:>7} tok  hit={c['hit_rate_median']}"
                  f"  ttft_p50={c['ttft_p50_median']}ms  runs={c['hit_rate_runs']}")


if __name__ == "__main__":
    main()
