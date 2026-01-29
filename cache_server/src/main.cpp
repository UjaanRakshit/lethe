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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
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
    // W7: Entry::local_data is an owned vector<byte> (not a borrowed
    // span). No lifetime gymnastics required — we can read it whenever
    // we want, and the proto's LookupResponse intentionally carries
    // no bytes anyway (just per-block metadata); the actual payload
    // goes back via Fetch / StreamBlocks. So local_data isn't even
    // referenced from here, which keeps the wire response cheap.
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
      // W7: respect the client's tier_hint. 0=HBM, 1=DRAM, 2=SSD.
      // Out-of-range or missing → default to DRAM (the always-on tier).
      const std::uint32_t raw_hint = b.tier_hint();
      kv.tier = (raw_hint <= 2)
                    ? static_cast<Tier>(raw_hint)
                    : Tier::DRAM;
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

    // W8: use the non-recursive FetchLocal path. The previous Lookup-
    // based implementation recursed: Fetch handler → cache_->Lookup
    // → read-repair (now firing post-W8 router fix) → FetchFromAny
    // → peer Fetch RPC → peer Lookup → peer read-repair → ... For a
    // cold-cache block this looped until every peer timed out, killing
    // throughput. FetchLocal only consults the local TieredStore.
    if (auto got = cache_->FetchLocal(id); got.has_value()) {
      resp->set_kv_data(got->data.data(), got->data.size());
      resp->set_found(true);
      resp->set_tier(static_cast<uint32_t>(got->tier));
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

namespace {

// Parse a comma-separated peer list in the format
// `node_id@host:port,node_id@host:port,...`. Empty input → empty
// vector (single-node mode). Entries equal to local_node_id are
// silently filtered (they'd point at ourselves).
std::vector<lethe::StaticPeer> ParsePeerList(const std::string& spec,
                                              const std::string& local_node_id) {
  std::vector<lethe::StaticPeer> out;
  if (spec.empty()) return out;
  std::istringstream iss(spec);
  std::string item;
  while (std::getline(iss, item, ',')) {
    if (item.empty()) continue;
    const auto at = item.find('@');
    if (at == std::string::npos) {
      std::cerr << "[lethe] --peers: skipping malformed entry "
                << item << " (expected node_id@host:port)\n";
      continue;
    }
    std::string node_id = item.substr(0, at);
    std::string address = item.substr(at + 1);
    if (node_id == local_node_id) continue;  // skip self
    out.push_back(lethe::StaticPeer{node_id, address});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // CLI: positional [node_id] [port] + optional --peers <spec>.
  // Format chosen for the run_3node.sh script's simplicity; argparse
  // would be heavier than this needs to be at W4 scope.
  lethe::CacheConfig cfg;
  cfg.node_id = (argc > 1) ? argv[1] : "node0";
  if (argc > 2) {
    cfg.grpc_port = static_cast<std::uint16_t>(std::stoi(argv[2]));
  }
  std::string peers_spec;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--peers" && i + 1 < argc) {
      peers_spec = argv[++i];
    } else if (a.rfind("--peers=", 0) == 0) {
      peers_spec = a.substr(8);
    }
  }
  cfg.seed_peers = ParsePeerList(peers_spec, cfg.node_id);

  // W1: keep capacity defaults small enough to fit on a laptop. The
  // hard rule is "default builds don't link ibverbs"; the soft choice
  // here is "default builds don't grab 32 GiB of host RAM either."
  cfg.dram_bytes = 1ULL << 30;   // 1 GiB
  // W7: enable a modest SSD tier by default so demotion paths get
  // exercised by smoke tests and the 3-node Python integration suite.
  // Per-node path keeps multiple lethe_servers on the same machine
  // from clobbering each other's SSD file.
  cfg.ssd_bytes = 256ULL << 20;  // 256 MiB
  cfg.ssd_path = std::string("/tmp/lethe-") + cfg.node_id + "/ssd";

  std::cout << "[lethe] node=" << cfg.node_id
            << " grpc=" << cfg.grpc_port
            << " dram=" << (cfg.dram_bytes >> 20) << "MiB"
            << " peers=" << cfg.seed_peers.size() << "\n";
  for (const auto& p : cfg.seed_peers) {
    std::cout << "[lethe]   peer " << p.node_id << " @ " << p.address << "\n";
  }

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
