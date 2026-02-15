"""Failover recovery curve: node kill -> full R=2 reconvergence (median of 3).

Process-based 3-node Lethe cluster on loopback (no docker needed), so it runs
anywhere the server binary builds. Routed R=2 insert, hard-kill one node
(SIGKILL), poll the survivors via local-only Fetch until every block is back
at R=2, record wall-clock from kill to full reconvergence; restart the victim
between reps. Median over REPS per working-set size.

PACE-loopback reference: 3.7 s @ 200 blocks ... 12.0 s @ 2000. The
docker-bridge variant is slower because of fixed per-RPC bridge latency; the
chaos harness covers that path.

Env: LETHE_SERVER_BIN (default build/cache_server/lethe_server under the repo).
"""
import hashlib
import os
import socket
import statistics
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "client"))

import logging  # noqa: E402

logging.getLogger("lethe_client.client").setLevel(logging.ERROR)
from lethe_client.client import BlockId, LetheClient  # noqa: E402
from lethe_client.routing import HashRing  # noqa: E402

BIN = os.environ.get(
    "LETHE_SERVER_BIN", os.path.join(_REPO, "build", "cache_server", "lethe_server"))
HOST = "127.0.0.1"
PORTS = [50051, 50052, 50053]
NODES = ["node0", "node1", "node2"]
PEERS = ",".join(f"{NODES[i]}@{HOST}:{PORTS[i]}" for i in range(3))
ADDRS = {NODES[i]: f"{HOST}:{PORTS[i]}" for i in range(3)}
PAYLOAD_LEN = 256
SIZES = [200, 500, 1000, 2000]
REPS = 3
REPLICATION_FACTOR = 2


def wait_port(port, timeout=20):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            with socket.create_connection((HOST, port), timeout=0.3):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def start_cluster():
    procs = {}
    for i in range(3):
        lf = open(f"/tmp/lethe-{NODES[i]}.log", "w")
        procs[NODES[i]] = subprocess.Popen(
            [BIN, NODES[i], str(PORTS[i]), "--peers", PEERS],
            stdout=lf, stderr=subprocess.STDOUT)
    for p in PORTS:
        if not wait_port(p):
            raise RuntimeError(f"node on {p} failed to bind")
    return procs


def stop_all(procs):
    for p in procs.values():
        try:
            p.kill()
        except Exception:
            pass
    for p in procs.values():
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def make_corpus(tag, n):
    c = {}
    for i in range(n):
        h = hashlib.sha256(f"{tag}-{i}".encode()).digest()
        c[h] = (hashlib.sha256(h).digest() * 8)[:PAYLOAD_LEN]
    return c


def insert_routed(corpus):
    ring = HashRing(NODES)
    groups = {}
    for h, p in corpus.items():
        primary = ring.route(h, n_replicas=1)[0]
        groups.setdefault(primary, []).append((BlockId(hash=h), p))
    for nid, blks in groups.items():
        with LetheClient(primary_address=ADDRS[nid], timeout_seconds=5) as c:
            c.insert(blks, request_id="recovery")


def holders(corpus, alive_nodes):
    clients = {n: LetheClient(primary_address=ADDRS[n], timeout_seconds=3)
               for n in alive_nodes}
    try:
        mn = 1 << 30
        for h, want in corpus.items():
            cnt = sum(1 for n in alive_nodes
                      if clients[n].fetch(BlockId(hash=h)) == want)
            mn = min(mn, cnt)
        return mn
    finally:
        for c in clients.values():
            c.close()


def measure_once(n, procs):
    corpus = make_corpus(f"rec-{n}-{time.time_ns()}", n)
    insert_routed(corpus)
    sample = dict(list(corpus.items())[:: max(1, n // 40)])
    t = time.monotonic()
    while time.monotonic() - t < 40:
        if holders(sample, NODES) >= REPLICATION_FACTOR:
            break
        time.sleep(0.5)
    victim = "node2"
    survivors = [x for x in NODES if x != victim]
    t0 = time.monotonic()
    procs[victim].kill()
    procs[victim].wait(timeout=5)
    conv = None
    while time.monotonic() - t0 < 180:
        if holders(sample, survivors) >= REPLICATION_FACTOR \
                and holders(corpus, survivors) >= REPLICATION_FACTOR:
            conv = time.monotonic() - t0
            break
        time.sleep(0.5)
    lf = open(f"/tmp/lethe-{victim}.log", "w")
    procs[victim] = subprocess.Popen(
        [BIN, victim, str(PORTS[2]), "--peers", PEERS],
        stdout=lf, stderr=subprocess.STDOUT)
    wait_port(PORTS[2])
    time.sleep(6)  # let rejoin re-replication settle before next rep
    return conv


def main():
    print(f"BIN={BIN}")
    procs = start_cluster()
    time.sleep(3)
    try:
        for n in SIZES:
            walls = [measure_once(n, procs) for _ in range(REPS)]
            good = [w for w in walls if w is not None]
            med = statistics.median(good) if good else float("nan")
            print(f"RECOVERY N={n} "
                  f"reps={[round(w, 1) if w is not None else 'NOCONV' for w in walls]} "
                  f"median={med:.1f}s")
    finally:
        stop_all(procs)


if __name__ == "__main__":
    main()
