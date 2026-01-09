// Lethe — cache server entry point.
//
// Parses flags, builds CacheConfig, instantiates LetheCache, stands up the
// gRPC server, optionally starts the RDMA transport, blocks on signal.

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

// gRPC
#include <grpcpp/grpcpp.h>
// #include "lethe.grpc.pb.h"   // generated from proto/lethe.proto

#include "lethe/cache.hpp"
#include "lethe/kv_transport.hpp"

// Transport factory: chooses GrpcStreamTransport (default) vs
// IbverbsTransport (when -DLETHE_ENABLE_RDMA=ON and the rxe device is
// usable). LetheCache owns the returned unique_ptr<KvTransport>.
//
// std::unique_ptr<lethe::KvTransport> MakeTransport(const lethe::CacheConfig& c) {
// #ifdef LETHE_ENABLE_RDMA
//   if (c.rdma_enabled && lethe::RdmaIsAvailable(c.rdma_device)) {
//     return std::make_unique<lethe::IbverbsTransport>(...);
//   }
// #endif
//   return std::make_unique<lethe::GrpcStreamTransport>(...);
// }

namespace {
std::atomic<bool> g_shutdown{false};
void HandleSignal(int) { g_shutdown.store(true); }
}  // namespace

// ---- gRPC service shim ----------------------------------------------------
//
// Thin adapter: deserialize the protobuf, call LetheCache, serialize the
// response. Keep all business logic in LetheCache so it stays testable
// without gRPC.
//
// class LetheServiceImpl final : public lethe::LetheCache::Service {
//   ...
// };
//
// TODO(W1): implement Lookup, Insert handlers.
// TODO(W3): wire Router into Lookup so RemoteHit responses include source.
// TODO(W4): implement StreamBlocks streaming server method.
// TODO(W8): implement Heartbeat, EvictBroadcast handlers. The Heartbeat
// shim is a strict field-by-field copy between HeartbeatReply and the
// proto HeartbeatResponse — no information loss in either direction:
//
//   ::grpc::Status Heartbeat(::grpc::ServerContext*,
//                            const HeartbeatRequest* req,
//                            HeartbeatResponse* resp) override {
//     auto reply = cache_->OnHeartbeat(req->node_id(), req->epoch());
//     for (const auto& peer : reply.alive_peers) {
//       auto* p = resp->add_alive_peers();
//       p->set_node_id(peer.node_id);
//       p->set_last_seen_epoch(peer.last_seen_epoch);
//       p->set_suspected(peer.suspected);
//     }
//     resp->set_cluster_epoch(reply.cluster_epoch);
//     return ::grpc::Status::OK;
//   }

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // TODO: replace with a real flag parser (gflags / CLI11).
  lethe::CacheConfig cfg;
  cfg.node_id = (argc > 1) ? argv[1] : "node0";
  if (argc > 2) cfg.grpc_port = static_cast<std::uint16_t>(std::stoi(argv[2]));
  // Seed peers passed as comma-separated list, argv[3].
  // RDMA toggle, argv[4].

  std::cout << "[lethe] node=" << cfg.node_id
            << " grpc=" << cfg.grpc_port << "\n";

  auto cache = std::make_unique<lethe::LetheCache>(cfg);
  cache->Start();

  // TODO(W1): build gRPC server bound to 0.0.0.0:cfg.grpc_port,
  // register LetheServiceImpl(cache.get()), call BuildAndStart().

  // TODO(W10): start metrics HTTP server on cfg.metrics_port.

  while (!g_shutdown.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::cout << "[lethe] shutting down\n";
  cache->Shutdown();
  return 0;
}
