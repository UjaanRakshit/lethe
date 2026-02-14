#!/usr/bin/env bash
# Lethe failover demo (W12.1) — the "hit rate survives a node kill" money shot,
# made visual on Grafana. This is W11's INV-5 driven against the live cluster.
#
# It reuses the chaos harness (chaos/invariants.py, scenario=sigkill): it
# inserts a corpus at R=2, hard-kills one node, and samples the ring-routed
# hit rate while the cluster detects the death and re-replicates. No bespoke
# demo wiring — the same code the chaos suite asserts against.
#
# Watch these dashboard panels at http://localhost:3000 (anonymous admin):
#   * Cluster epoch         — steps up ~3.3 s after the kill (heartbeat
#                             detection; dead_after=3 s). The clearest signal.
#   * Under-replicated blocks — spikes to the victim's in-route block count,
#                             then drains back to 0 as survivors re-replicate.
#   * Cache hit rate        — stays high through the kill: the survivors still
#                             hold the R=2 replicas, so the server-side hit
#                             ratio does NOT crash. That is INV-5, the load
#                             path surviving a node loss.
#
# Note: the chaos run ALSO prints its own client-side ring hit-rate, which dips
# (~1.00 -> ~0.75) during the ~3 s detection window — lookups aimed at the dead
# node before the client's ring re-routes — and recovers. That dip is in the
# scenario's stdout (INV-5: min hit-rate > 0), not on the Grafana ratio panel
# (those client-side misses never reach a live server to be counted).
#
# Verified run (pure cache, no GPU): baseline ring hit-rate 1.00 -> min 0.75 ->
# recover; epoch bumped at t+3.5s; 48 blocks reconverged to R=2 by t+7.7s; all
# six invariants PASS.
#
# Usage:  bash scripts/failover_demo.sh [scenario]   # default: sigkill
#         (try `restart` to also see the victim rejoin and re-converge)
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COMPOSE="docker compose -f deploy/docker-compose.yml"
SCENARIO="${1:-sigkill}"

# lethe_client on the path: prefer a venv, else the in-repo client/ package.
VENV="${LETHE_VENV:-$ROOT/.venv-vllm}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  # Relative path resolves against the cwd ($ROOT) on POSIX and Windows alike;
  # an absolute git-bash path (/c/...) is not understood by a Windows python.
  export PYTHONPATH="client${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "[demo] bringing up a fresh 3-node cluster + Prometheus + Grafana..."
$COMPOSE down >/dev/null 2>&1
$COMPOSE up -d >/dev/null 2>&1
for _ in $(seq 1 60); do
  ok=1
  for p in 9091 9092 9093; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$p/metrics" 2>/dev/null)
    [[ "$code" == "200" ]] || ok=0
  done
  [[ $ok -eq 1 ]] && break
  sleep 0.5
done

echo "[demo] Grafana: http://localhost:3000  (dashboard: Lethe — Distributed KV Cache)"
echo "[demo] watch: Cache hit rate (dips, recovers, never 0) | Cluster epoch (bumps)"
echo "[demo]        | Under-replicated blocks (spikes, drains to 0)"
echo "[demo] running chaos scenario '$SCENARIO' in 5s — keep the dashboard open..."
sleep 5

python -m chaos.invariants --scenario "$SCENARIO"
rc=$?

echo
echo "[demo] done (stack left up for inspection; tear down: $COMPOSE down)"
exit $rc
