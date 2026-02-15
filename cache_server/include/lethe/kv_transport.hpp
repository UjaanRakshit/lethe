#pragma once
// Bulk KV transport for block transfer.
//
// gRPC handles control-plane RPCs and small messages. For KV block transfer
// (the bytes that matter for throughput) we want RDMA over SoftRoCE (rxe)
// when available, falling back to a gRPC bidi-stream transport otherwise.
// Both sit behind the abstract KvTransport interface defined here.
//
// Concrete implementations:
//   - GrpcStreamTransport (always available; default when LETHE_ENABLE_RDMA
//     is OFF or RdmaIsAvailable() returns false).
//   - IbverbsTransport (ibverbs / SoftRoCE; built only when
//     LETHE_ENABLE_RDMA=ON). Queue pair per peer, established lazily on
//     first Send; pinned buffer pool per peer for sends, shared pool for
//     receives.
//
// The interface is implementation-agnostic so that swapping the data path
// is a constructor change in main.cpp.

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

struct RdmaConfig {
  std::string device_name;          // e.g. "rxe0"
  std::uint16_t port = 1;
  std::uint16_t listen_port = 50052;
  std::size_t send_buffer_bytes = 16ULL << 20;   // 16 MiB per peer
  std::size_t recv_buffer_bytes = 64ULL << 20;   // 64 MiB shared
  std::size_t max_inflight_per_peer = 16;
};

// Abstract bulk-transport interface. No data members; all state lives in
// the derived classes. Owners hold this by std::unique_ptr<KvTransport>.
//
// Two block-shaped operations are exposed:
//   - Send: push semantics. Caller owns the bytes; transport stages /
//           transmits / ACKs. Used for replication push and prefill→decode.
//   - Fetch: pull semantics. Caller asks a peer for a known BlockId;
//            transport returns the materialized block (or nullopt on
//            miss / failure). Used for read-repair.
//
// Both return std::future so an ibverbs implementation can dispatch onto
// its completion-polling thread without blocking the caller; the gRPC
// implementation runs the work synchronously and returns a ready future.
// Replicator's bounded queue + worker pool decides how many concurrent
// Send/Fetch calls are in flight at once; the transport doesn't impose
// its own concurrency cap.
class KvTransport {
 public:
  using OnReceiveFn = std::function<void(BlockId, std::vector<std::byte>,
                                          StreamPurpose)>;

  virtual ~KvTransport() = default;

  virtual void Start() = 0;
  virtual void Shutdown() = 0;

  // Establish a connection to a peer. Idempotent.
  virtual void Connect(const std::string& peer_id,
                       const std::string& address) = 0;
  virtual void Disconnect(const std::string& peer_id) = 0;

  // Send a single block. The future completes when the peer ACKs.
  // Zero-copy from `data` is NOT guaranteed; the implementation may stage
  // the block through a pinned buffer pool.
  virtual std::future<bool> Send(const std::string& peer_id,
                                 BlockId id,
                                 StreamPurpose purpose,
                                 std::span<const std::byte> data) = 0;

  // Fetch a single block from a peer by id. Returns the materialized block
  // on hit, nullopt on miss or RPC failure. Per-call deadline is the
  // implementation's choice (gRPC: 500ms; ibverbs: TBD when hardware
  // arrives). Distinct from Send semantically — Fetch is request/response;
  // Send is fire-and-stage.
  virtual std::future<std::optional<KvBlock>> Fetch(
      const std::string& peer_id, BlockId id) = 0;
};

// gRPC bidi-stream transport. Always available. Default when RDMA is off,
// and currently the actual data path. Send invokes Insert RPCs (replication
// push); Fetch invokes the Fetch RPC. Both run synchronously inside the call
// and return a ready future — true async dispatch happens at the
// Replicator's worker-pool layer above us. gRPC is the data path despite
// the "RDMA for KV transfer" wording in DESIGN.md §6.
class GrpcStreamTransport : public KvTransport {
 public:
  GrpcStreamTransport(RdmaConfig cfg, OnReceiveFn on_receive);
  ~GrpcStreamTransport() override;
  void Start() override;
  void Shutdown() override;
  void Connect(const std::string& peer_id,
               const std::string& address) override;
  void Disconnect(const std::string& peer_id) override;
  std::future<bool> Send(const std::string& peer_id,
                         BlockId id,
                         StreamPurpose purpose,
                         std::span<const std::byte> data) override;
  std::future<std::optional<KvBlock>> Fetch(
      const std::string& peer_id, BlockId id) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// libibverbs / SoftRoCE transport. Header-only declaration unless
// LETHE_ENABLE_RDMA=ON AND a SoftRoCE / real-IB device is present at
// runtime. This could not be validated against real hardware (WSL2 kernel
// ships without the InfiniBand subsystem); the abstraction stays so the
// swap is a constructor change in main.cpp when hardware arrives.
class IbverbsTransport : public KvTransport {
 public:
  IbverbsTransport(RdmaConfig cfg, OnReceiveFn on_receive);
  ~IbverbsTransport() override;
  void Start() override;
  void Shutdown() override;
  void Connect(const std::string& peer_id,
               const std::string& address) override;
  void Disconnect(const std::string& peer_id) override;
  std::future<bool> Send(const std::string& peer_id,
                         BlockId id,
                         StreamPurpose purpose,
                         std::span<const std::byte> data) override;
  std::future<std::optional<KvBlock>> Fetch(
      const std::string& peer_id, BlockId id) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Helper: query SoftRoCE availability. Used by main.cpp to gate RDMA path.
bool RdmaIsAvailable(const std::string& device_name);

}  // namespace lethe
