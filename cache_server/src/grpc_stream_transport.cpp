// Lethe — GrpcStreamTransport. Default bulk-KV transport; the production
// data path for W5-6 since SoftRoCE is unavailable in the WSL2 dev
// environment (see docs/decisions/W5_rdma_fallback.md).
//
// Wire model:
//   * Send  → Insert RPC with a single Block. Replication push reaches
//             the peer's gRPC Insert handler → LetheCache::Insert → local
//             tier write. Replicator's bounded queue + worker pool decides
//             how many concurrent Sends fly at once; this transport runs
//             each Send synchronously inside the worker thread and
//             resolves a ready future, so the worker is naturally
//             back-pressured by its own thread budget.
//   * Fetch → Fetch RPC, single block. Same synchronous-inside-call shape
//             as Send; the parallel-fan-out concurrency for read-repair
//             is the caller's responsibility (Replicator::FetchFromAny
//             uses std::async per peer).
//
// Why not StreamBlocks (the bidi-stream RPC defined in proto/lethe.proto)?
// Insert is what W4's Replicator used, and the existing 3-node smoke
// asserts replication via Insert's wire path. Swapping to StreamBlocks
// here would force the W4 server-side handler to grow a bidi-stream
// implementation, which is scope creep we don't need for the fallback
// session. The W12 IB-hardware swap is a transport-level change, not a
// wire-level change.
//
// OnReceive callback: this transport has no recv-side responsibility.
// Receive happens through the regular gRPC service handlers in main.cpp
// (Insert / Fetch / StreamBlocks RPC entry points). The OnReceiveFn
// is stored but not invoked here; the field exists to keep the
// constructor signature symmetric with IbverbsTransport (which DOES
// need a recv callback when it lands at W12 — RDMA receives don't
// flow through the gRPC handlers).

#include "lethe/kv_transport.hpp"

#include <chrono>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

#include <grpcpp/grpcpp.h>

#include "lethe.grpc.pb.h"
#include "lethe.pb.h"
#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr int kSendDeadlineMs = 2000;
constexpr int kFetchDeadlineMs = 500;

::lethe::rpc::BlockId BlockIdToProto(const BlockId& src) {
  ::lethe::rpc::BlockId p;
  p.set_hash(src.hash.data(), src.hash.size());
  p.set_layer(src.layer);
  p.set_head_group(src.head_group);
  p.set_model_id(src.model_id);
  return p;
}

BlockId BlockIdFromProto(const ::lethe::rpc::BlockId& src) {
  BlockId out;
  const std::string& h = src.hash();
  const std::size_t n = std::min<std::size_t>(h.size(), out.hash.size());
  std::memcpy(out.hash.data(), h.data(), n);
  if (n < out.hash.size()) {
    std::memset(out.hash.data() + n, 0, out.hash.size() - n);
  }
  out.layer = src.layer();
  out.head_group = src.head_group();
  out.model_id = src.model_id();
  return out;
}

struct PeerStub {
  std::string address;
  std::shared_ptr<grpc::Channel> channel;
  std::unique_ptr<::lethe::rpc::LetheCache::Stub> stub;
};

}  // namespace

struct GrpcStreamTransport::Impl {
  RdmaConfig cfg;
  OnReceiveFn on_receive;

  std::mutex mu;
  std::unordered_map<std::string, PeerStub> peers;

  ::lethe::rpc::LetheCache::Stub* stub_for(const std::string& peer_id) {
    std::lock_guard<std::mutex> g(mu);
    auto it = peers.find(peer_id);
    return it == peers.end() ? nullptr : it->second.stub.get();
  }
};

GrpcStreamTransport::GrpcStreamTransport(RdmaConfig cfg, OnReceiveFn on_receive)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = std::move(cfg);
  impl_->on_receive = std::move(on_receive);
  // The on_receive callback is stored for symmetry with IbverbsTransport
  // but is never invoked here — gRPC receives are dispatched by main.cpp's
  // service handlers, not by us. Cast-to-void quiets the unused-warn.
  (void)impl_->on_receive;
}

GrpcStreamTransport::~GrpcStreamTransport() = default;

void GrpcStreamTransport::Start() {
  // Channels open lazily on Connect. Nothing to do here.
}

void GrpcStreamTransport::Shutdown() {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->peers.clear();
}

void GrpcStreamTransport::Connect(const std::string& peer_id,
                                  const std::string& address) {
  std::lock_guard<std::mutex> g(impl_->mu);
  auto it = impl_->peers.find(peer_id);
  if (it != impl_->peers.end()) return;
  PeerStub p;
  p.address = address;
  // Insecure for the W5-6 single-rack / loopback scope. TLS is a W11+
  // chaos-suite concern; deferred.
  p.channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  p.stub = ::lethe::rpc::LetheCache::NewStub(p.channel);
  impl_->peers.emplace(peer_id, std::move(p));
}

void GrpcStreamTransport::Disconnect(const std::string& peer_id) {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->peers.erase(peer_id);
}

std::future<bool> GrpcStreamTransport::Send(
    const std::string& peer_id,
    BlockId id,
    StreamPurpose /*purpose*/,
    std::span<const std::byte> data) {
  std::promise<bool> pr;
  auto fut = pr.get_future();

  auto* stub = impl_->stub_for(peer_id);
  if (stub == nullptr) {
    pr.set_value(false);
    return fut;
  }

  // Build the Insert request and dispatch synchronously. The future is
  // resolved before we return, which is fine — the caller (Replicator's
  // worker) is already a background thread.
  ::lethe::rpc::InsertRequest req;
  req.set_request_id("");
  req.set_source_node("");  // populated by Replicator via a separate path
                            // when wire-tagging by source becomes useful
                            // (W8 will surface this).
  auto* b = req.add_blocks();
  *b->mutable_id() = BlockIdToProto(id);
  b->set_kv_data(data.data(), data.size());
  b->set_tier_hint(static_cast<std::uint32_t>(Tier::DRAM));

  ::lethe::rpc::InsertResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(kSendDeadlineMs));
  auto status = stub->Insert(&ctx, req, &resp);
  pr.set_value(status.ok() && resp.accepted_count() > 0);
  return fut;
}

std::future<std::optional<KvBlock>> GrpcStreamTransport::Fetch(
    const std::string& peer_id, BlockId id) {
  std::promise<std::optional<KvBlock>> pr;
  auto fut = pr.get_future();

  auto* stub = impl_->stub_for(peer_id);
  if (stub == nullptr) {
    pr.set_value(std::nullopt);
    return fut;
  }

  ::lethe::rpc::FetchRequest req;
  *req.mutable_id() = BlockIdToProto(id);
  req.set_requesting_node("lethe-fetch");
  ::lethe::rpc::FetchResponse resp;
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(kFetchDeadlineMs));
  auto status = stub->Fetch(&ctx, req, &resp);
  if (!status.ok() || !resp.found()) {
    pr.set_value(std::nullopt);
    return fut;
  }
  KvBlock blk;
  blk.id = BlockIdFromProto(resp.id());
  const std::string& d = resp.kv_data();
  blk.data.resize(d.size());
  std::memcpy(blk.data.data(), d.data(), d.size());
  blk.tier = Tier::DRAM;
  pr.set_value(std::move(blk));
  return fut;
}

// Free function declared in kv_transport.hpp. Always-built TU so this
// symbol is present regardless of LETHE_ENABLE_RDMA. The current body
// reports "no" unconditionally — main.cpp's transport factory does its
// own LETHE_ENABLE_RDMA #ifdef gate to decide whether to try
// IbverbsTransport. When real IB hardware lands (W12 PACE), this body
// becomes a libibverbs probe (open device, check capabilities).
bool RdmaIsAvailable(const std::string& /*device_name*/) {
  return false;
}

}  // namespace lethe
