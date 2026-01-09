#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# C++ cache server.
cmake -B build -S . \
    -DLETHE_ENABLE_RDMA="${LETHE_ENABLE_RDMA:-OFF}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# Python client.
pip install -e client

# Regenerate gRPC stubs.
python -m grpc_tools.protoc \
    -Iproto \
    --python_out=client/lethe_client \
    --grpc_python_out=client/lethe_client \
    proto/lethe.proto

echo "Build complete."
