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

FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libprotobuf32 libgrpc++1 libibverbs1 librdmacm1 \
    iptables iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/cache_server/lethe_server /usr/local/bin/lethe_server
WORKDIR /var/lib/lethe
ENTRYPOINT ["/usr/local/bin/lethe_server"]
