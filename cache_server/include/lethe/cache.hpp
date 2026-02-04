#pragma once
// Lethe — top-level cache coordinator.
//
// Owns the lifetimes of every subsystem (block store, routing, replication,
// eviction, membership, RDMA, metrics) and exposes the small surface area
// that the gRPC service handlers actually call.
//
// Threading: all public methods are thread-safe. The facade itself holds no
// state lock — each subsystem owns its own mutex and a Cache method that
// touches N subsystems acquires their locks in declaration order.
//
// vLLM integration: the Python-side KV transfer connector lives in
// client/lethe_client/vllm_hook.py. That file is the ONLY place that talks
// to the vLLM scheduler; everything below is provider-agnostic.

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

// Forward declarations to keep this header light.
class TieredStore;
class Router;
class Replicator;
class Evictor;
class Membership;
class KvTransport;
class Metrics;

struct CacheConfig {
  std::string node_id;
  // Static peer set (W3-W4: immutable for process lifetime; W8 makes
  // this dynamic via heartbeat-driven membership). Each entry binds a
  // routing identity (node_id) to a transport address (host:port).
  std::vector<StaticPeer> seed_peers;

  // Capacities per tier (bytes).
  std::size_t hbm_bytes = 0;                 // set to 0 to disable HBM tier
  std::size_t dram_bytes = 32ULL << 30;      // default 32 GiB
  std::size_t ssd_bytes = 256ULL << 30;      // default 256 GiB
  std::string ssd_path = "/var/lib/lethe";

  // Distribution.
  std::uint32_t replication_factor = 2;
  std::uint32_t virtual_nodes_per_peer = 128;

  // Transport.
  bool rdma_enabled = false;
  std::uint16_t grpc_port = 50051;
  std::uint16_t rdma_port = 50052;
  std::string rdma_device;                   // e.g. "rxe0" for SoftRoCE

  // Observability.
  std::uint16_t metrics_port = 9090;

  // Eviction.
  std::size_t eviction_high_watermark_pct = 90;
  std::size_t eviction_low_watermark_pct = 75;
};

// Result of a Lookup() against the local view of the cluster.
struct LookupResult {
  struct Entry {
    BlockId id;
    enum class Where { LocalHit, RemoteHit, Miss };
    Where where = Where::Miss;
    std::optional<std::string> remote_node;  // populated when RemoteHit
    Tier tier = Tier::DRAM;                  // for LocalHit / RemoteHit
    // For LocalHit: an OWNED copy of the block payload (W7 change from
    // W1's borrowed span). The SSD tier can't safely lend spans into
    // mmap'd memory across slot reuse, and uniform ownership is simpler
    // than per-tier dispatch. Cost: one memcpy per Get. Negligible at
    // 64 KiB block sizes. Empty when not LocalHit.
    std::vector<std::byte> local_data;
  };
  std::vector<Entry> entries;
  std::uint32_t hit_count = 0;
  std::uint32_t miss_count = 0;
};

// Options for Insert. Default is async replication: the call returns once
// the primary tier write completes; replica pushes happen on a background
// thread and any failures show up in `lethe_replicas_under_target`. Setting
// `sync_replicate = true` blocks the call until R-1 replicas ACK (or
// timeout). Use sync only for tests and for the durability-critical writes
// in the chaos suite; the production / benchmark path is async.
struct InsertOptions {
  bool sync_replicate = false;
  std::chrono::milliseconds sync_timeout{500};
};

class LetheCache {
 public:
  explicit LetheCache(CacheConfig cfg);
  ~LetheCache();

  LetheCache(const LetheCache&) = delete;
  LetheCache& operator=(const LetheCache&) = delete;

  // -- RPC handlers ---------------------------------------------------------

  LookupResult Lookup(const std::vector<BlockId>& ids,
                      const std::string& request_id,
                      const std::string& requesting_node);

  // Returns the number of blocks accepted into the local tier (not the
  // number of replicas ACKed). Replication semantics: see InsertOptions.
  std::uint32_t Insert(std::vector<KvBlock> blocks,
                       const std::string& request_id,
                       const std::string& source_node,
                       InsertOptions opts = {});

  void IngestStreamedBlock(BlockId id,
                           std::vector<std::byte> payload,
                           StreamPurpose purpose);

  // Local-only fetch: returns block bytes + tier if present in the
  // local TieredStore, nullopt otherwise. Does NOT consult the router
  // and does NOT trigger read-repair. This is what the gRPC Fetch
  // handler uses — going through Lookup would recurse:
  // Fetch handler → Lookup → read-repair → FetchFromAny → peer Fetch
  // → peer Lookup → peer read-repair → ... So Fetch needs a non-
  // recursive entry point. Same data shape as a LocalHit entry's
  // local_data / tier fields.
  struct LocalFetchResult {
    std::vector<std::byte> data;
    Tier tier;
  };
  std::optional<LocalFetchResult> FetchLocal(const BlockId& id);

  HeartbeatReply OnHeartbeat(const std::string& peer_id,
                             std::uint64_t peer_epoch);

  void OnEvictBroadcast(const std::vector<BlockId>& evicted,
                        const std::string& source);

  // -- Lifecycle ------------------------------------------------------------

  void Start();
  void Shutdown();

  // -- Introspection (for tests + metrics) ----------------------------------

  // Delegates to Membership — the membership epoch is the single source of
  // truth; the facade does not maintain its own copy.
  std::uint64_t cluster_epoch() const noexcept;
  const std::string& node_id() const noexcept { return cfg_.node_id; }

 private:
  CacheConfig cfg_;
  std::unique_ptr<TieredStore> store_;
  std::unique_ptr<Router> router_;
  // Metrics is a leaf that Replicator/Membership/Evictor record into
  // from their worker/heartbeat/eviction threads. It MUST outlive all
  // of them, so it is declared HERE (before transport_/replicator_/…)
  // → destroyed AFTER them in reverse-declaration order. Its own dtor
  // joins the /metrics HTTP thread, which only touches its own atomics.
  std::unique_ptr<Metrics> metrics_;
  // Subsystem destruction order matters and is encoded by declaration
  // order (reverse-decl is destruction order):
  //   1. evictor_ destroyed first → its 3 tier threads join while
  //      Membership + TieredStore are still alive (the threads dereference
  //      both during a final scan pass that may already be in flight).
  //   2. membership_ next → its heartbeat thread joins while Router +
  //      Replicator are still alive (HeartbeatLoop calls Router::SetPeers
  //      and Replicator::TriggerReReplication on a final membership tick).
  //   3. replicator_ next → its worker pool joins while transport_ is
  //      still alive (workers call transport_->Send).
  //   4. transport_ next → channels close.
  //   5. router_ and store_ last.
  //
  // Declaration order below puts the LATEST-destroyed members first;
  // do NOT re-order without re-reasoning about the worker-join → peer
  // subsystem teardown chain.
  std::unique_ptr<KvTransport> transport_;
  std::unique_ptr<Replicator> replicator_;
  std::unique_ptr<Membership> membership_;
  std::unique_ptr<Evictor> evictor_;

  std::atomic<bool> running_{false};
};

}  // namespace lethe
