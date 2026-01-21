#!/usr/bin/env bash
# Bring up a 3-node Lethe cluster locally on loopback ports
# 50051 / 50052 / 50053. Each node knows about the other two via
# the same --peers spec (the local entry is filtered server-side).
#
# Cleanup: writes the three child PIDs to a tempfile and traps EXIT
# (covers normal exit, Ctrl-C, and most error paths) to kill them.
# Background server processes have their stdout/stderr redirected
# to /tmp/lethe-node{0,1,2}.log; if anything looks wrong, those are
# the first place to look.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${ROOT}/build/cache_server/lethe_server"

if [[ ! -x "${BIN}" ]]; then
  echo "lethe_server not built at ${BIN}" >&2
  echo "  → run: bash scripts/build.sh" >&2
  exit 1
fi

PORTS=(50051 50052 50053)
NODES=(node0 node1 node2)
HOST=127.0.0.1

# Build the comma-separated peer spec shared by all three nodes.
# Each node's --peers includes ALL three entries; the server filters
# out its own node_id at parse time.
PEERS_SPEC=""
for i in 0 1 2; do
  [[ -n "${PEERS_SPEC}" ]] && PEERS_SPEC+=","
  PEERS_SPEC+="${NODES[$i]}@${HOST}:${PORTS[$i]}"
done

PIDFILE="$(mktemp -t lethe-3node.XXXXXX)"
LOGS=()
PIDS=()

cleanup() {
  echo "[run_3node] cleaning up..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -INT "${pid}" 2>/dev/null
    fi
  done
  # Give them a moment to shut down cleanly.
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null
    fi
  done
  for pid in "${PIDS[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
  rm -f "${PIDFILE}"
  echo "[run_3node] done."
}
trap cleanup EXIT INT TERM

for i in 0 1 2; do
  LOG="/tmp/lethe-${NODES[$i]}.log"
  LOGS+=("${LOG}")
  echo "[run_3node] starting ${NODES[$i]} on ${HOST}:${PORTS[$i]}, log=${LOG}"
  "${BIN}" "${NODES[$i]}" "${PORTS[$i]}" --peers "${PEERS_SPEC}" \
      > "${LOG}" 2>&1 &
  PIDS+=($!)
  echo "$!" >> "${PIDFILE}"
done

# Poll each node's port until it accepts a TCP connection (or fail
# after a few seconds). gRPC takes a moment to bind.
for i in 0 1 2; do
  for attempt in $(seq 1 50); do
    if (exec 3<>/dev/tcp/${HOST}/${PORTS[$i]}) 2>/dev/null; then
      exec 3<&-; exec 3>&-
      echo "[run_3node]   ${NODES[$i]} bound (${attempt} polls)"
      break
    fi
    sleep 0.1
    if [[ ${attempt} -eq 50 ]]; then
      echo "[run_3node] ERROR: ${NODES[$i]} failed to bind ${PORTS[$i]} after 5s" >&2
      echo "[run_3node] log tail:" >&2
      tail -20 "${LOGS[$i]}" >&2
      exit 1
    fi
  done
done

echo "[run_3node] cluster up. PIDs in ${PIDFILE}, logs in /tmp/lethe-node{0,1,2}.log"
echo "[run_3node] peer spec: ${PEERS_SPEC}"
echo "[run_3node] Ctrl-C to stop."

# Block. The trap above handles cleanup on SIGINT/SIGTERM/EXIT.
wait
