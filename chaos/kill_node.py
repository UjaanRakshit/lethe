"""Chaos harness — kill nodes in a running Lethe cluster (W11).

Usage:
    python -m chaos.kill_node \
        --target lethe-node1 \
        --mode sigkill \
        --restart-after 5

Modes:
  - sigkill: docker kill, immediate.
  - sigterm: graceful shutdown.
  - pause: docker pause (simulates a long GC pause / network partition).

After the kill (and optional restart), the orchestrator should observe:
  - other nodes' Lookups for blocks owned by the killed node return
    RemoteHit pointing at a replica (W4 read-repair).
  - cluster_epoch increments (W8 membership change).
  - re-replication completes within REPLICATION_RECOVERY_BUDGET seconds.
  - failover demo measured by chaos.invariants assertions.
"""

from __future__ import annotations

import argparse
import subprocess
import time


def kill(target: str, mode: str) -> None:
    if mode == "sigkill":
        subprocess.check_call(["docker", "kill", "-s", "KILL", target])
    elif mode == "sigterm":
        subprocess.check_call(["docker", "kill", "-s", "TERM", target])
    elif mode == "pause":
        subprocess.check_call(["docker", "pause", target])
    else:
        raise ValueError(f"unknown mode {mode}")


def revive(target: str, mode: str) -> None:
    if mode == "pause":
        subprocess.check_call(["docker", "unpause", target])
    else:
        subprocess.check_call(["docker", "start", target])


def is_running(target: str) -> bool:
    """True iff the container exists and is in the `running` state. A paused
    container reports `paused`, a sigkilled/stopped one `exited` — both are
    not-running for our purposes (the restore step start/unpauses them)."""
    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", target],
        capture_output=True,
        text=True,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, help="docker container name")
    p.add_argument("--mode", choices=["sigkill", "sigterm", "pause"], default="sigkill")
    p.add_argument(
        "--restart-after",
        type=float,
        default=0,
        help="seconds after kill to restart; 0 = leave dead",
    )
    args = p.parse_args()

    print(f"[chaos] killing {args.target} ({args.mode})")
    t0 = time.monotonic()
    kill(args.target, args.mode)
    if args.restart_after > 0:
        time.sleep(args.restart_after)
        revive(args.target, args.mode)
        print(f"[chaos] revived {args.target} after {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
