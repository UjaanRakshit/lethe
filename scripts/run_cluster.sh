#!/usr/bin/env bash
# Bring up the 3-node Lethe cluster locally via docker-compose.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[lethe] building image..."
docker compose -f deploy/docker-compose.yml build

echo "[lethe] starting cluster + Prometheus + Grafana..."
docker compose -f deploy/docker-compose.yml up -d

echo
echo "Cluster running:"
echo "  node0:  localhost:50051   metrics: localhost:9091"
echo "  node1:  localhost:50061   metrics: localhost:9092"
echo "  node2:  localhost:50071   metrics: localhost:9093"
echo "  prom:   localhost:9090"
echo "  grafana:localhost:3000"
echo
echo "Tear down:  docker compose -f deploy/docker-compose.yml down"
