"""Inject configurable packet loss via tc/netem (W11)."""
from __future__ import annotations
import argparse, subprocess, time


def loss(target: str, pct: float, latency_ms: float = 0):
    cmd = ["docker", "exec", target, "tc", "qdisc", "add", "dev", "eth0",
           "root", "netem", "loss", f"{pct}%"]
    if latency_ms > 0:
        cmd += ["delay", f"{latency_ms}ms"]
    subprocess.check_call(cmd)


def clear(target: str):
    subprocess.check_call(["docker", "exec", target, "tc", "qdisc", "del",
                           "dev", "eth0", "root"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--loss-pct", type=float, default=5)
    p.add_argument("--latency-ms", type=float, default=0)
    p.add_argument("--duration", type=float, default=30)
    args = p.parse_args()
    loss(args.target, args.loss_pct, args.latency_ms)
    try:
        time.sleep(args.duration)
    finally:
        clear(args.target)


if __name__ == "__main__":
    main()
