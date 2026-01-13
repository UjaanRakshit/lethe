#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Python interpreter override. Default: `python3` on PATH. Set PYTHON=
# in the environment to force a specific one (useful when the system
# `python3` is a partial build, e.g. compiled without ssl).
PYTHON="${PYTHON:-python3}"

# C++ cache server.
cmake -B build -S . \
    -DLETHE_ENABLE_RDMA="${LETHE_ENABLE_RDMA:-OFF}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# Python client.
"${PYTHON}" -m pip install -e client

# Regenerate gRPC stubs.
"${PYTHON}" -m grpc_tools.protoc \
    -Iproto \
    --python_out=client/lethe_client \
    --grpc_python_out=client/lethe_client \
    --pyi_out=client/lethe_client \
    proto/lethe.proto

# protoc-gen-grpc-python emits a top-level `import lethe_pb2` which
# breaks when lethe_pb2 lives in the lethe_client package. Rewrite to
# the relative form so the stub is importable post-install. This is
# the standard workaround for this long-standing grpcio-tools quirk.
sed -i 's/^import lethe_pb2 as/from . import lethe_pb2 as/' \
    client/lethe_client/lethe_pb2_grpc.py

echo "Build complete."
