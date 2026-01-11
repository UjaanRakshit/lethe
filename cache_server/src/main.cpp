// Lethe — cache server entry point.
//
// Builds a CacheConfig, instantiates LetheCache, stands up the gRPC
// server, blocks on a shutdown signal.
//
// The gRPC service implementation lives at the bottom of this file as
// per cache_server/CLAUDE.md ("gRPC service implementations live in
// src/main.cpp for now; split when they grow past ~300 lines"). All
// the actual business logic delegates to LetheCache; the shim is a
// straight deserialize → call → serialize loop with no policy in it.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include "lethe.grpc.pb.h"
#include "lethe.pb.h"
#include "lethe/cache.hpp"
#include "lethe/kv_transport.hpp"

namespace {
std::atomic<bool> g_shutdown{false};
void HandleSignal(int) { g_shutdown.store(true); }

// Transport factory: chooses GrpcStreamTransport (default) vs
// IbverbsTransport (when -DLETHE_ENABLE_RDMA=ON and the rxe device is
// usable). LetheCache owns the returned unique_ptr<KvTransport>.
//
// W1: this factory is unused — the cache doesn't take a KvTransport
// yet (no Replicator). W4+ wires it in when replication needs a
// non-control-plane bulk transport.
}  // namespace

// ---------------------------------------------------------------------------
// gRPC service shim
// ---------------------------------------------------------------------------

namespace lethe {

namespace {

// Copy a wire-format BlockId (proto) into our in-memory BlockId.
BlockId BlockIdFromProto(const ::lethe::rpc::BlockId& src) {
  BlockId out;
  const std::string& h = src.hash();
  // Defensive: clamp to 32 bytes; over- or under-length is a client
  // bug, but we'd rather truncate than UB.
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

void BlockIdToProto(const BlockId& src, ::lethe::rpc::BlockId* dst) {
  dst->set_hash(src.hash.data(), src.hash.size());
  dst->set_layer(src.layer);
  dst->set_head_group(src.head_group);
  dst->set_model_id(src.model_id);
}

}  // namespace

class LetheServiceImpl final : public ::lethe::rpc::LetheCache::Service {
 public:
  explicit LetheServiceImpl(LetheCache* cache) : cache_(cache) {}

  grpc::Status Lookup(grpc::ServerContext* /*ctx*/,
                      const ::lethe::rpc::LookupRequest* req,
                      ::lethe::rpc::LookupResponse* resp) override {
    std::vector<BlockId> ids;
    ids.reserve(req->block_ids_size());
    for (const auto& pb_id : req->block_ids()) {
      ids.push_back(BlockIdFromProto(pb_id));
    }
    LookupResult result =
        cache_->Lookup(ids, req->request_id(), req->requesting_node());
    // CONTRACT (cache.hpp:81-86): Entry::local_data spans are valid
    // only until the next mutating cache call on the same BlockId.
    // We serialize them into the response NOW — before this method
    // returns and before any concurrent Insert can land. The proto's
    // LookupResponse intentionally carries no bytes (just metadata);
    // actual bytes go via Fetch / StreamBlocks. So the local_data is
    // not even referenced from here, which is the simplest possible
    // way to honor the contract.
    for (const auto& entry : result.entries) {
      if (entry.where == LookupResult::Entry::Where::LocalHit) {
        auto* hit = resp->add_hits();
        BlockIdToProto(entry.id, hit->mutable_id());
        hit->set_source_node(cache_->node_id());
        hit->set_tier(static_cast<uint32_t>(entry.tier));
      } else if (entry.where == LookupResult::Entry::Where::RemoteHit) {
        auto* hit = resp->add_hits();
        BlockIdToProto(entry.id, hit->mutable_id());
        hit->set_source_node(
            entry.remote_node ? *entry.remote_node : std::string{});
        hit->set_tier(static_cast<uint32_t>(entry.tier));
      } else {
        BlockIdToProto(entry.id, resp->add_misses());
      }
    }
    return grpc::Status::OK;
  }

  grpc::Status Insert(grpc::ServerContext* /*ctx*/,
                      const ::lethe::rpc::InsertRequest* req,
                      ::lethe::rpc::InsertResponse* resp) override {
    std::vector<KvBlock> blocks;
    blocks.reserve(req->blocks_size());
    for (const auto& b : req->blocks()) {
      KvBlock kv;
      kv.id = BlockIdFromProto(b.id());
      const std::string& payload = b.kv_data();
      kv.data.resize(payload.size());
      std::memcpy(kv.data.data(), payload.data(), payload.size());
      kv.tier = Tier::DRAM;
      blocks.push_back(std::move(kv));
    }
    const std::uint32_t accepted = cache_->Insert(
        std::move(blocks), req->request_id(), req->source_node());
    resp->set_accepted_count(accepted);
    resp->set_rejected_count(
        static_cast<std::uint32_t>(req->blocks_size()) - accepted);
    return grpc::Status::OK;
  }

  grpc::Status Fetch(grpc::ServerContext* /*ctx*/,
                     const ::lethe::rpc::FetchRequest* req,
                     ::lethe::rpc::FetchResponse* resp) override {
    BlockId id = BlockIdFromProto(req->id());
    BlockIdToProto(id, resp->mutable_id());

    // Lookup one block; if LocalHit, we have a span into the store —
    // copy it into the response payload immediately (contract).
    LookupResult result = cache_->Lookup({id}, /*request_id=*/{}, /*requesting_node=*/{});
    if (result.entries.size() == 1 &&
        result.entries[0].where == LookupResult::Entry::Where::LocalHit) {
      const auto& span = result.entries[0].local_data;
      resp->set_kv_data(span.data(), span.size());
      resp->set_found(true);
      resp->set_tier(static_cast<uint32_t>(result.entries[0].tier));
    } else {
      resp->set_found(false);
    }
    return grpc::Status::OK;
  }

  grpc::Status StreamBlocks(
      grpc::ServerContext* /*ctx*/,
      grpc::ServerReader<::lethe::rpc::BlockChunk>* reader,
      ::lethe::rpc::StreamAck* resp) override {
    // Client-streaming: peer pushes blocks at us. W1 collects them
    // into LetheCache::IngestStreamedBlock. The chunked-large-block
    // reassembly path (offset/last fields) lands in W4 when read-repair
    // streams big payloads.
    ::lethe::rpc::BlockChunk chunk;
    std::uint32_t received = 0;
    std::uint64_t total_bytes = 0;
    while (reader->Read(&chunk)) {
      BlockId id = BlockIdFromProto(chunk.id());
      std::vector<std::byte> payload(chunk.payload().size());
      std::memcpy(payload.data(), chunk.payload().data(), payload.size());
      total_bytes += payload.size();
      StreamPurpose purpose = StreamPurpose::ReplicationPush;
      switch (chunk.purpose()) {
        case ::lethe::rpc::BlockChunk::REPLICATION:
          purpose = StreamPurpose::ReplicationPush; break;
        case ::lethe::rpc::BlockChunk::READ_REPAIR:
          purpose = StreamPurpose::ReadRepair; break;
        case ::lethe::rpc::BlockChunk::PREFILL_TO_DECODE:
          purpose = StreamPurpose::Promotion; break;
        default: break;
      }
      cache_->IngestStreamedBlock(id, std::move(payload), purpose);
      ++received;
    }
    resp->set_received_count(received);
    resp->set_total_bytes(total_bytes);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext* /*ctx*/,
                         const ::lethe::rpc::HeartbeatRequest* req,
                         ::lethe::rpc::HeartbeatResponse* resp) override {
    HeartbeatReply reply = cache_->OnHeartbeat(req->node_id(), req->epoch());
    for (const auto& peer : reply.alive_peers) {
      auto* p = resp->add_alive_peers();
      p->set_node_id(peer.node_id);
      p->set_last_seen_epoch(peer.last_seen_epoch);
      p->set_suspected(peer.suspected);
    }
    resp->set_cluster_epoch(reply.cluster_epoch);
    return grpc::Status::OK;
  }

  grpc::Status EvictBroadcast(
      grpc::ServerContext* /*ctx*/,
      const ::lethe::rpc::EvictBroadcastRequest* req,
      ::lethe::rpc::EvictBroadcastResponse* resp) override {
    std::vector<BlockId> evicted;
    evicted.reserve(req->evicted_size());
    for (const auto& pb_id : req->evicted()) {
      evicted.push_back(BlockIdFromProto(pb_id));
    }
    cache_->OnEvictBroadcast(evicted, req->source_node());
    resp->set_ack(true);
    return grpc::Status::OK;
  }

 private:
  LetheCache* cache_;  // not owned
};

}  // namespace lethe

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  lethe::CacheConfig cfg;
  cfg.node_id = (argc > 1) ? argv[1] : "node0";
  if (argc > 2) {
    cfg.grpc_port = static_cast<std::uint16_t>(std::stoi(argv[2]));
  }
  // W3+ : seed peers via argv[3] (comma-separated).
  // W5-6: RDMA toggle via argv[4].

  // W1: keep capacity defaults small enough to fit on a laptop. The
  // hard rule is "default builds don't link ibverbs"; the soft choice
  // here is "default builds don't grab 32 GiB of host RAM either."
  cfg.dram_bytes = 1ULL << 30;   // 1 GiB
  cfg.ssd_bytes = 0;             // SSD tier disabled in W1.

  std::cout << "[lethe] node=" << cfg.node_id
            << " grpc=" << cfg.grpc_port
            << " dram=" << (cfg.dram_bytes >> 20) << "MiB\n";

  auto cache = std::make_unique<lethe::LetheCache>(cfg);
  cache->Start();

  lethe::LetheServiceImpl service(cache.get());

  grpc::ServerBuilder builder;
  const std::string addr = "0.0.0.0:" + std::to_string(cfg.grpc_port);
  builder.AddListeningPort(addr, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  if (!server) {
    std::cerr << "[lethe] failed to bind " << addr << "\n";
    cache->Shutdown();
    return 1;
  }
  std::cout << "[lethe] gRPC listening on " << addr << "\n";

  // Block on shutdown signal; periodic wake so SIGINT registers.
  while (!g_shutdown.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::cout << "[lethe] shutting down\n";
  server->Shutdown(std::chrono::system_clock::now() +
                   std::chrono::milliseconds(500));
  cache->Shutdown();
  return 0;
}
