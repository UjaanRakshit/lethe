#!/usr/bin/env bash
# SoftRoCE setup for the RDMA path.
#
# SoftRoCE (rxe) provides RDMA semantics over a regular Ethernet NIC, so you
# can develop ibverbs code without IB hardware; the API is identical.
#
# Tested on Ubuntu 22.04 / 24.04 with kernels >= 5.15.

set -euo pipefail

IFACE="${1:-eth0}"

echo "[lethe] loading rdma_rxe..."
sudo modprobe rdma_rxe

echo "[lethe] enumerating existing rxe devices..."
rdma link show || true

if ! rdma link show | grep -q "rxe0"; then
    echo "[lethe] adding rxe0 on $IFACE..."
    sudo rdma link add rxe0 type rxe netdev "$IFACE"
fi

echo "[lethe] rxe is up:"
rdma link show
ibv_devinfo -d rxe0 | head -n 20

echo
echo "[lethe] smoke test: ibv_rc_pingpong in two terminals:"
echo "    server:  ibv_rc_pingpong -d rxe0 -g 0"
echo "    client:  ibv_rc_pingpong -d rxe0 -g 0 <server-ip>"
