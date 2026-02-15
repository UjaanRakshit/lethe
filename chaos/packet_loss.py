"""Inject configurable packet loss into a Lethe node.

Preferred mechanism is tc/netem. But the WSL2 Docker kernel ships without
`sch_netem` (no module, no modprobe), so `tc qdisc add ... netem` fails with
"Specified qdisc kind is unknown". We therefore fall back to an iptables
`statistic` random-drop rule on INPUT+OUTPUT, which IS available there and
drops the same fraction of packets in both directions. netem is still tried
first so this works unchanged on a kernel that has it.
"""

from __future__ import annotations

import argparse
import subprocess
import time


def _netem_loss(target: str, pct: float, latency_ms: float) -> bool:
    cmd = [
        "docker",
        "exec",
        target,
        "tc",
        "qdisc",
        "add",
        "dev",
        "eth0",
        "root",
        "netem",
        "loss",
        f"{pct}%",
    ]
    if latency_ms > 0:
        cmd += ["delay", f"{latency_ms}ms"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def _iptables_loss(target: str, pct: float) -> bool:
    prob = max(0.0, min(1.0, pct / 100.0))
    ok = True
    for chain in ("INPUT", "OUTPUT"):
        r = subprocess.run(
            [
                "docker",
                "exec",
                target,
                "iptables",
                "-A",
                chain,
                "-m",
                "statistic",
                "--mode",
                "random",
                "--probability",
                f"{prob:.6f}",
                "-j",
                "DROP",
            ],
            capture_output=True,
            text=True,
        )
        ok = ok and r.returncode == 0
    return ok


def loss(target: str, pct: float, latency_ms: float = 0) -> str:
    """Drop ~pct% of `target`'s packets. Returns the mechanism used
    ("netem" or "iptables"). Raises if neither mechanism is available."""
    if _netem_loss(target, pct, latency_ms):
        return "netem"
    if latency_ms > 0:
        print(
            f"[chaos] netem unavailable; iptables fallback ignores "
            f"latency_ms={latency_ms}"
        )
    if _iptables_loss(target, pct):
        return "iptables"
    raise RuntimeError(
        f"could not inject packet loss on {target}: neither tc/netem nor "
        "iptables statistic is available"
    )


def clear(target: str) -> None:
    """Remove any injected loss (both mechanisms). Idempotent / best-effort:
    deleting an absent qdisc/rule is not an error, so invariants.py can call
    clear() unconditionally in a finally / restore step."""
    subprocess.run(
        ["docker", "exec", target, "tc", "qdisc", "del", "dev", "eth0", "root"],
        capture_output=True,
        text=True,
    )
    # Drop every statistic-random DROP rule by listing and deleting by line
    # number (handles any probability value, and multiple accumulated rules).
    for chain in ("INPUT", "OUTPUT"):
        listing = subprocess.run(
            ["docker", "exec", target, "iptables", "-L", chain, "--line-numbers", "-n"],
            capture_output=True,
            text=True,
        )
        lines = [ln for ln in listing.stdout.splitlines() if "statistic" in ln]
        # Delete from highest line number down so earlier indices stay valid.
        for ln in reversed(lines):
            num = ln.split()[0]
            if num.isdigit():
                subprocess.run(
                    ["docker", "exec", target, "iptables", "-D", chain, num],
                    capture_output=True,
                    text=True,
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--loss-pct", type=float, default=5)
    p.add_argument("--latency-ms", type=float, default=0)
    p.add_argument("--duration", type=float, default=30)
    args = p.parse_args()
    mech = loss(args.target, args.loss_pct, args.latency_ms)
    print(
        f"[chaos] {args.loss_pct}% loss on {args.target} via {mech} "
        f"for {args.duration}s"
    )
    try:
        time.sleep(args.duration)
    finally:
        clear(args.target)
        print("[chaos] cleared")


if __name__ == "__main__":
    main()
