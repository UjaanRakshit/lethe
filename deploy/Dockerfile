# Lethe cache server image.
FROM ubuntu:24.04 AS build

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git pkg-config \
    libprotobuf-dev protobuf-compiler \
    libgrpc++-dev protobuf-compiler-grpc \
    libibverbs-dev librdmacm-dev \
    iptables iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    && cmake --build build -j$(nproc)

# Runtime deps. We install the -dev gRPC/protobuf packages rather than
# the slim runtime ones (libgrpc++1 etc.): on Ubuntu 24.04 the runtime
# package soname doesn't match what the build stage linked against
# (the binary wants libgrpc++.so.1.51), so the slim image exited 127 on
# a missing shared object. The dev packages guarantee the exact .so is
# present. Slightly larger image; correct, and this is a dev/demo
# deploy, not a shipped artifact.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libprotobuf-dev libgrpc++-dev libibverbs1 librdmacm1 \
    iptables iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/cache_server/lethe_server /usr/local/bin/lethe_server
WORKDIR /var/lib/lethe
ENTRYPOINT ["/usr/local/bin/lethe_server"]
