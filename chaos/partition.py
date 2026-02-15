"""Network partition injection via iptables.

Drops packets between two nodes for a configurable duration to simulate
a split-brain scenario. Lethe should NOT proceed to elect a new owner
(we don't run consensus); instead, both sides should mark each other
suspected and resume normal operation when packets flow again.
"""

from __future__ import annotations
import argparse
import subprocess
import time


def partition(a: str, b: str):
    # Bidirectional drop between two container hostnames.
    subprocess.check_call(
        [
            "docker",
            "exec",
            a,
            "iptables",
            "-A",
            "OUTPUT",
            "-d",
            b,
            "-j",
            "DROP",
        ]
    )
    subprocess.check_call(
        [
            "docker",
            "exec",
            b,
            "iptables",
            "-A",
            "OUTPUT",
            "-d",
            a,
            "-j",
            "DROP",
        ]
    )


def heal(a: str, b: str):
    # Idempotent / best-effort: `iptables -D` exits non-zero if the rule is
    # absent (already healed, or never injected). Deleting a missing rule is
    # not an error for us, so swallow it — this lets invariants.py call heal()
    # unconditionally in a finally / restore step.
    for src, dst in ((a, b), (b, a)):
        subprocess.run(
            [
                "docker",
                "exec",
                src,
                "iptables",
                "-D",
                "OUTPUT",
                "-d",
                dst,
                "-j",
                "DROP",
            ],
            capture_output=True,
            text=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--duration", type=float, default=10)
    args = p.parse_args()
    print(f"[chaos] partitioning {args.a} <-X-> {args.b} for {args.duration}s")
    partition(args.a, args.b)
    try:
        time.sleep(args.duration)
    finally:
        heal(args.a, args.b)
        print("[chaos] healed")


if __name__ == "__main__":
    main()
