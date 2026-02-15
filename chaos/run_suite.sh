#!/usr/bin/env bash
# Lethe chaos suite. Brings up the 3-node docker cluster, then runs five
# failure-injection scenarios sequentially and asserts the cluster
# invariants (chaos/invariants.py). Target wall-clock: < 5 minutes.
#
# Exit code 0 iff every scenario passes. On failure, exits 1 and the final
# line names exactly which scenario:invariant pairs failed.
#
# Prereqs:
#   * the lethe:dev image is built  (docker compose -f deploy/docker-compose.yml build)
#   * a venv with the lethe_client package on PATH (.venv-vllm by default)
#   * docker with NET_ADMIN-capable containers (compose grants it)
#
# The five scenarios (see chaos/invariants.py for the invariant defs):
#   1. sigkill      — hard-kill a node, leave dead; verify INV-1..6.
#   2. restart      — hard-kill, restart, verify clean rejoin + reconverge.
#   3. pause        — long SIGSTOP (>dead_after), unpause, verify rejoin.
#   4. partition    — iptables split node1<-X->node2; no split-brain, heal.
#   5. packet_loss  — 5% netem on a node; no false death, load path alive.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COMPOSE="docker compose -f deploy/docker-compose.yml"
VENV="${LETHE_VENV:-$ROOT/.venv-vllm}"
SCENARIOS=(sigkill restart pause partition packet_loss large)
SUITE_BUDGET_S=300
KEEP_UP="${KEEP_UP:-1}"     # 1 = leave the stack up for inspection

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "[suite] WARNING: venv not found at $VENV; relying on system python" >&2
fi

start_ts=$(date +%s)

# Each scenario starts from a FRESH cluster (down clears the in-memory block
# stores). Blocks are never deleted, so a reused store accumulates across
# scenarios; because re-replication re-pushes EVERY in-route block a node
# holds (idempotent over-push — it can't know which a peer is missing without
# ACKs), a polluted store would inflate each scenario's recovery time and make
# the size-aware INV-3 budget unfair. A clean store per scenario keeps recovery
# proportional to that scenario's own corpus. The LARGE scenario deliberately
# exceeds the kBoundedScan=256 cap.
fresh_cluster() {
  $COMPOSE down >/dev/null 2>&1
  $COMPOSE up -d >/dev/null 2>&1
  for _ in $(seq 1 60); do
    ok=1
    for p in 9091 9092 9093; do
      code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$p/metrics" 2>/dev/null)
      [[ "$code" == "200" ]] || ok=0
    done
    [[ $ok -eq 1 ]] && return 0
    sleep 0.5
  done
  return 1
}

declare -a FAILED
overall=0
for s in "${SCENARIOS[@]}"; do
  elapsed=$(( $(date +%s) - start_ts ))
  if (( elapsed > SUITE_BUDGET_S )); then
    echo "[suite] ERROR: exceeded ${SUITE_BUDGET_S}s budget before scenario '$s'" >&2
    FAILED+=("$s:BUDGET")
    overall=1
    break
  fi

  echo
  echo "######## scenario: $s (t+${elapsed}s) ########"
  if ! fresh_cluster; then
    echo "[suite] ERROR: cluster did not become ready for '$s'" >&2
    FAILED+=("$s:SETUP")
    overall=1
    continue
  fi
  # Each scenario is its own process + fresh cluster; invariants.py also runs
  # its own restore() in a finally so a mid-scenario abort still cleans up.
  out="$(python -m chaos.invariants --scenario "$s" 2>&1)"
  rc=$?
  echo "$out"
  # Pull the machine-readable RESULT line for the aggregate report.
  result_line="$(echo "$out" | grep -E '^RESULT scenario=' | tail -1)"
  failed_invs="$(echo "$result_line" | sed -n 's/.*failed=\([^ ]*\).*/\1/p')"
  if [[ $rc -ne 0 || ( -n "$failed_invs" && "$failed_invs" != "none" ) ]]; then
    overall=1
    if [[ -n "$failed_invs" && "$failed_invs" != "none" ]]; then
      IFS=',' read -ra invs <<< "$failed_invs"
      for inv in "${invs[@]}"; do FAILED+=("$s:$inv"); done
    else
      FAILED+=("$s:ERROR(rc=$rc)")
    fi
  fi
done

total=$(( $(date +%s) - start_ts ))
echo
echo "================ chaos suite summary ================"
echo "wall-clock: ${total}s (budget ${SUITE_BUDGET_S}s)"
if (( overall == 0 )); then
  echo "RESULT: ALL PASS (${#SCENARIOS[@]} scenarios)"
else
  echo "RESULT: FAIL — ${FAILED[*]}"
fi

if [[ "$KEEP_UP" == "1" ]]; then
  echo "(stack left up; tear down with: $COMPOSE down)"
else
  $COMPOSE down >/dev/null 2>&1 || true
fi

exit $overall
