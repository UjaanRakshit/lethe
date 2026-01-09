#pragma once
// Lethe — bulk KV transport for block transfer (W5–6).
//
// gRPC handles control-plane RPCs and small messages. For KV block transfer
// (the bytes that actually matter for throughput), we want RDMA over
// SoftRoCE (rxe) when available, and fall back to a gRPC bidi-stream
// transport otherwise. Both implementations sit behind the abstract
// KvTransport interface defined here.
//
// Concrete implementations:
//   - GrpcStreamTransport (always available; default when LETHE_ENABLE_RDMA
//     is OFF or RdmaIsAvailable() returns false).
//   - IbverbsTransport (ibverbs / SoftRoCE; built only when
//     LETHE_ENABLE_RDMA=ON). Queue pair per peer, established lazily on
//     first Send; pinned buffer pool per peer for sends, shared pool for
//     receives.
//
// SCOPE GUARDRAIL: if SoftRoCE setup or QP debugging exceeds 2.5 weeks of
// W5–6, the fallback is to ship the gRPC transport and call it done. The
// interface here is implementation-agnostic so that swap is a constructor
// change in main.cpp.

#include <cstddef>
#include <functional>
#include <future>
#include <memory>
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
};

// gRPC bidi-stream transport. Always available. Default when RDMA is off.
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
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// libibverbs / SoftRoCE transport. Compiled only when LETHE_ENABLE_RDMA=ON.
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
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Helper: query SoftRoCE availability. Used by main.cpp to gate RDMA path.
bool RdmaIsAvailable(const std::string& device_name);

}  // namespace lethe
