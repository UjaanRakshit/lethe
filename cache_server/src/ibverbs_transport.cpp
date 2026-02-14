// Lethe — IbverbsTransport (compiled only when LETHE_ENABLE_RDMA=ON).
//
// W12.2: real implementation against InfiniBand verbs (PACE ICE, ConnectX-7).
// Replaces the W5-6 abort-stub now that real IB hardware is available.
//
// Wire model (deliberately the simplest correct shape under the 6h time box):
//   * Send  → two-sided RDMA SEND/RECV. The push path (replication +
//             re-replication + prefill→decode) is the throughput-critical
//             path, so it gets RDMA. The sender copies [MsgHeader|payload]
//             into a registered buffer and ibv_post_send; the receiver's
//             pre-posted RECV completes, we parse the header and hand the
//             bytes to on_receive() (which cache.cpp wires to LetheCache).
//   * Fetch → delegated to an internal GrpcStreamTransport. Read-repair pull
//             is occasional, and answering a fetch over RDMA would need the
//             block store inside the transport (it isn't here). Reusing the
//             tested gRPC Fetch keeps correctness without duplicating proto
//             code. The RDMA win is on Send, which is what Phase 3 measures.
//
// Connection: rdma_cm (librdmacm). IPoIB is present on PACE, so
// rdma_resolve_addr/route derive the IB path/GID/MTU for us — far less
// silent-hang surface than a hand-built QP state machine on InfiniBand.
// Each node listens (rdma_listen) and accepts inbound connections (one
// accepted QP per peer, used to RECV that peer's Sends), and dials each peer
// (one outbound QP per peer, used to SEND to it). One completion-polling
// thread per connection (CLAUDE.md threading invariant).
//
// Addressing: rdma_cm needs IB-fabric IPs (ib0), not the management
// hostnames the gRPC control plane uses. Endpoints come from env:
//   LETHE_RDMA_LISTEN = "<ib0_ip>:<port>"            (this node)
//   LETHE_RDMA_PEERS  = "id@<ib0_ip>:<port>,..."     (peers)
// set by the launch scripts. Connect(peer_id, grpc_addr) ignores grpc_addr
// for the RDMA path and looks peer_id up in the env map (it still forwards
// to the gRPC fallback for Fetch).

#include "lethe/kv_transport.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lethe/types.hpp"

namespace lethe {

namespace {

constexpr std::uint32_t kMagic = 0x4C455448;  // "LETH"
constexpr std::uint32_t kMsgBlockPush = 1;
constexpr int kCqDepth = 128;
constexpr int kMaxRecvBuffers = 16;   // pre-posted RECVs per accepted QP
constexpr int kMaxSendBuffers = 8;    // send ring per outbound QP
// Each registered buffer is kHeaderBytes + msg_bytes, and ibv_reg_mr PINS it.
// (recv+send)*conns*procs buffers add up fast, so the default is modest; the
// Phase-3 large-block microbench raises it via LETHE_RDMA_MSG_BYTES.
constexpr std::size_t kDefaultMsgBytes = 1ULL << 20;  // 1 MiB

// Fixed wire header. Trailing payload (KV bytes) follows in the same buffer.
#pragma pack(push, 1)
struct MsgHeader {
  std::uint32_t magic;
  std::uint32_t type;
  std::uint32_t purpose;
  std::uint32_t payload_len;
  std::uint8_t  hash[32];
  std::uint32_t layer;
  std::uint32_t head_group;
  std::uint32_t model_id;
};
#pragma pack(pop)

constexpr std::size_t kHeaderBytes = sizeof(MsgHeader);

void Warn(const char* what, int err) {
  std::fprintf(stderr, "[lethe][rdma] %s failed: %s (%d)\n", what,
               std::strerror(err > 0 ? err : errno), err);
}

// Parse "id@ip:port,id@ip:port" into a map id -> (ip, port).
std::unordered_map<std::string, std::pair<std::string, std::uint16_t>>
ParsePeerEnv(const char* raw) {
  std::unordered_map<std::string, std::pair<std::string, std::uint16_t>> out;
  if (!raw) return out;
  std::string s(raw);
  std::size_t i = 0;
  while (i < s.size()) {
    std::size_t comma = s.find(',', i);
    std::string tok = s.substr(i, comma == std::string::npos ? comma : comma - i);
    auto at = tok.find('@');
    auto colon = tok.rfind(':');
    if (at != std::string::npos && colon != std::string::npos && colon > at) {
      std::string id = tok.substr(0, at);
      std::string ip = tok.substr(at + 1, colon - at - 1);
      std::uint16_t port = static_cast<std::uint16_t>(
          std::stoi(tok.substr(colon + 1)));
      out[id] = {ip, port};
    }
    if (comma == std::string::npos) break;
    i = comma + 1;
  }
  return out;
}

struct RegBuf {
  std::vector<std::byte> mem;
  ibv_mr* mr = nullptr;
};

// One RC connection: either an outbound QP (we Send on it) or an inbound
// accepted QP (we RECV on it). rdma_cm owns the id/qp/pd; we own the buffers
// and the poll thread.
struct Conn {
  rdma_cm_id* id = nullptr;       // owns qp, pd via rdma_create_qp
  ibv_pd* pd = nullptr;
  ibv_cq* cq = nullptr;
  ibv_comp_channel* chan = nullptr;
  bool inbound = false;           // accepted (RECV side) vs dialed (SEND side)
  std::string peer_id;

  // Inbound: pre-posted recv buffers, keyed by wr_id == buffer index.
  std::vector<RegBuf> recv_bufs;
  // Outbound: send ring + free list + wr_id -> promise.
  std::vector<RegBuf> send_bufs;
  std::vector<int> free_send;
  std::mutex send_mu;
  std::map<std::uint64_t, std::shared_ptr<std::promise<bool>>> pending;
  std::uint64_t next_wr = 1;

  std::thread poller;
  std::atomic<bool> stop{false};
};

}  // namespace

struct IbverbsTransport::Impl {
  RdmaConfig cfg;
  OnReceiveFn on_receive;

  std::string listen_ip;
  std::uint16_t listen_port = 0;
  std::size_t msg_bytes = kDefaultMsgBytes;  // per-buffer payload capacity
  std::unordered_map<std::string, std::pair<std::string, std::uint16_t>> peers;

  // Fetch + gRPC-side Connect delegate to a real gRPC transport.
  std::unique_ptr<GrpcStreamTransport> grpc_fallback;

  rdma_event_channel* listen_ch = nullptr;
  rdma_cm_id* listen_id = nullptr;
  std::thread listen_thread;
  std::atomic<bool> stop{false};

  // Diagnostics (W12.2 bring-up): dumped on Shutdown.
  std::atomic<std::uint64_t> n_posted{0}, n_done_ok{0}, n_done_err{0};
  std::atomic<std::uint64_t> n_drop_noconn{0}, n_drop_bp{0}, n_recv{0};

  std::mutex conns_mu;
  std::mutex dial_mu;  // serializes lazy dials so two Sends don't double-dial
  std::unordered_map<std::string, std::unique_ptr<Conn>> outbound;  // peer_id -> dialed
  std::vector<std::unique_ptr<Conn>> inbound;                       // accepted QPs

  // Lazily dial a peer on first Send (gRPC connects lazily too; an eager dial
  // at Connect() races a 3-node simultaneous startup where peers aren't
  // listening yet). Returns the outbound Conn* or nullptr if unreachable.
  Conn* EnsureOutbound(const std::string& peer_id) {
    {
      std::lock_guard<std::mutex> g(conns_mu);
      auto it = outbound.find(peer_id);
      if (it != outbound.end()) return it->second.get();
    }
    std::lock_guard<std::mutex> dg(dial_mu);
    {
      std::lock_guard<std::mutex> g(conns_mu);  // double-check after dial_mu
      auto it = outbound.find(peer_id);
      if (it != outbound.end()) return it->second.get();
    }
    auto pit = peers.find(peer_id);
    if (pit == peers.end()) return nullptr;
    auto c = Dial(peer_id, pit->second.first, pit->second.second);  // slow; no conns_mu
    if (!c) return nullptr;
    std::lock_guard<std::mutex> g(conns_mu);
    auto [it, _] = outbound.emplace(peer_id, std::move(c));
    return it->second.get();
  }

  // ---- buffer + QP helpers ------------------------------------------------

  std::size_t MsgCap() const { return kHeaderBytes + msg_bytes; }

  bool MakeQp(rdma_cm_id* id, Conn* c) {
    c->id = id;
    c->pd = ibv_alloc_pd(id->verbs);
    if (!c->pd) { Warn("ibv_alloc_pd", 0); return false; }
    c->chan = nullptr;  // busy-poll; no event channel needed
    c->cq = ibv_create_cq(id->verbs, kCqDepth, nullptr, nullptr, 0);
    if (!c->cq) { Warn("ibv_create_cq", 0); return false; }
    ibv_qp_init_attr qa{};
    qa.send_cq = c->cq;
    qa.recv_cq = c->cq;
    qa.qp_type = IBV_QPT_RC;
    qa.cap.max_send_wr = kMaxSendBuffers + 4;
    qa.cap.max_recv_wr = kMaxRecvBuffers + 4;
    qa.cap.max_send_sge = 1;
    qa.cap.max_recv_sge = 1;
    if (rdma_create_qp(id, c->pd, &qa)) { Warn("rdma_create_qp", 0); return false; }
    return true;
  }

  RegBuf MakeBuf(ibv_pd* pd) {
    RegBuf b;
    b.mem.resize(MsgCap());
    b.mr = ibv_reg_mr(pd, b.mem.data(), b.mem.size(),
                      IBV_ACCESS_LOCAL_WRITE);
    if (!b.mr) Warn("ibv_reg_mr", 0);
    return b;
  }

  bool PostRecv(Conn* c, int idx) {
    ibv_sge sge{};
    sge.addr = reinterpret_cast<std::uintptr_t>(c->recv_bufs[idx].mem.data());
    sge.length = static_cast<std::uint32_t>(c->recv_bufs[idx].mem.size());
    sge.lkey = c->recv_bufs[idx].mr->lkey;
    ibv_recv_wr wr{};
    wr.wr_id = static_cast<std::uint64_t>(idx);   // recv wr_id == buffer index
    wr.sg_list = &sge;
    wr.num_sge = 1;
    ibv_recv_wr* bad = nullptr;
    int rc = ibv_post_recv(c->id->qp, &wr, &bad);
    if (rc) { Warn("ibv_post_recv", rc); return false; }
    return true;
  }

  // Inbound side: register + post recv buffers, start poller.
  void SetupInbound(Conn* c) {
    for (int i = 0; i < kMaxRecvBuffers; ++i) {
      c->recv_bufs.push_back(MakeBuf(c->pd));
      PostRecv(c, i);
    }
    c->poller = std::thread([this, c] { PollInbound(c); });
  }

  // Outbound side: register send buffers, start poller (for send completions).
  void SetupOutbound(Conn* c) {
    for (int i = 0; i < kMaxSendBuffers; ++i) {
      c->send_bufs.push_back(MakeBuf(c->pd));
      c->free_send.push_back(i);
    }
    c->poller = std::thread([this, c] { PollOutbound(c); });
  }

  void PollInbound(Conn* c) {
    ibv_wc wc[16];
    while (!c->stop.load(std::memory_order_relaxed)) {
      int n = ibv_poll_cq(c->cq, 16, wc);
      if (n < 0) { Warn("ibv_poll_cq(in)", 0); break; }
      for (int i = 0; i < n; ++i) {
        int idx = static_cast<int>(wc[i].wr_id);
        if (wc[i].status != IBV_WC_SUCCESS) {
          // On a disconnect the flushed recvs surface here; don't re-post.
          continue;
        }
        if (wc[i].opcode == IBV_WC_RECV) {
          HandleRecv(c, idx, wc[i].byte_len);
          PostRecv(c, idx);   // recycle the buffer
        }
      }
      if (n == 0) std::this_thread::yield();
    }
  }

  void PollOutbound(Conn* c) {
    ibv_wc wc[16];
    while (!c->stop.load(std::memory_order_relaxed)) {
      int n = ibv_poll_cq(c->cq, 16, wc);
      if (n < 0) { Warn("ibv_poll_cq(out)", 0); break; }
      for (int i = 0; i < n; ++i) {
        std::uint64_t wr = wc[i].wr_id;
        bool ok = wc[i].status == IBV_WC_SUCCESS;
        std::shared_ptr<std::promise<bool>> pr;
        int buf_idx = -1;
        {
          std::lock_guard<std::mutex> g(c->send_mu);
          auto it = c->pending.find(wr);
          if (it != c->pending.end()) {
            pr = it->second;
            buf_idx = static_cast<int>(wr & 0xffff);  // low bits == buf index
            c->pending.erase(it);
            c->free_send.push_back(buf_idx);
          }
        }
        if (pr) pr->set_value(ok);
        if (ok) n_done_ok.fetch_add(1); else { n_done_err.fetch_add(1); Warn("send completion", wc[i].status); }
      }
      if (n == 0) std::this_thread::yield();
    }
  }

  void HandleRecv(Conn* c, int idx, std::uint32_t byte_len) {
    if (byte_len < kHeaderBytes) return;
    const auto* h = reinterpret_cast<const MsgHeader*>(c->recv_bufs[idx].mem.data());
    if (h->magic != kMagic || h->type != kMsgBlockPush) return;
    if (kHeaderBytes + h->payload_len > byte_len) return;
    BlockId id;
    std::memcpy(id.hash.data(), h->hash, 32);
    id.layer = h->layer;
    id.head_group = h->head_group;
    id.model_id = h->model_id;
    std::vector<std::byte> payload(h->payload_len);
    std::memcpy(payload.data(), c->recv_bufs[idx].mem.data() + kHeaderBytes,
                h->payload_len);
    n_recv.fetch_add(1);
    if (on_receive) {
      on_receive(id, std::move(payload),
                 static_cast<StreamPurpose>(h->purpose));
    }
  }

  // ---- rdma_cm connection handshakes -------------------------------------

  void ListenLoop() {
    while (!stop.load(std::memory_order_relaxed)) {
      rdma_cm_event* ev = nullptr;
      if (rdma_get_cm_event(listen_ch, &ev)) break;  // channel closed on stop
      auto type = ev->event;
      rdma_cm_id* cid = ev->id;
      if (type == RDMA_CM_EVENT_CONNECT_REQUEST) {
        auto c = std::make_unique<Conn>();
        c->inbound = true;
        if (MakeQp(cid, c.get())) {
          SetupInbound(c.get());
          rdma_conn_param cp{};
          cp.responder_resources = 1;
          cp.initiator_depth = 1;
          cp.retry_count = 7;
          cp.rnr_retry_count = 7;  // retry on receiver-not-ready, don't error the QP
          if (rdma_accept(cid, &cp)) Warn("rdma_accept", 0);
          cid->context = c.get();
          std::lock_guard<std::mutex> g(conns_mu);
          inbound.push_back(std::move(c));
        }
      } else if (type == RDMA_CM_EVENT_DISCONNECTED) {
        // Best-effort: stop the poller for that conn; buffers freed on Shutdown.
        auto* c = static_cast<Conn*>(cid->context);
        if (c) c->stop.store(true);
        rdma_disconnect(cid);
      }
      rdma_ack_cm_event(ev);
    }
  }

  // Client-side blocking handshake to a peer ib0 endpoint.
  std::unique_ptr<Conn> Dial(const std::string& peer_id,
                             const std::string& ip, std::uint16_t port) {
    rdma_event_channel* ch = rdma_create_event_channel();
    if (!ch) { Warn("rdma_create_event_channel", 0); return nullptr; }
    rdma_cm_id* cid = nullptr;
    if (rdma_create_id(ch, &cid, nullptr, RDMA_PS_TCP)) {
      Warn("rdma_create_id", 0); rdma_destroy_event_channel(ch); return nullptr;
    }
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &dst.sin_addr);

    auto await = [&](rdma_cm_event_type want) -> bool {
      rdma_cm_event* ev = nullptr;
      if (rdma_get_cm_event(ch, &ev)) return false;
      bool ok = ev->event == want;
      if (!ok) std::fprintf(stderr, "[lethe][rdma] dial: got event %d want %d\n",
                            ev->event, want);
      rdma_ack_cm_event(ev);
      return ok;
    };

    if (rdma_resolve_addr(cid, nullptr, reinterpret_cast<sockaddr*>(&dst), 2000)) {
      Warn("rdma_resolve_addr", 0); return nullptr;
    }
    if (!await(RDMA_CM_EVENT_ADDR_RESOLVED)) return nullptr;
    if (rdma_resolve_route(cid, 2000)) { Warn("rdma_resolve_route", 0); return nullptr; }
    if (!await(RDMA_CM_EVENT_ROUTE_RESOLVED)) return nullptr;

    auto c = std::make_unique<Conn>();
    c->inbound = false;
    c->peer_id = peer_id;
    if (!MakeQp(cid, c.get())) return nullptr;
    SetupOutbound(c.get());

    rdma_conn_param cp{};
    cp.responder_resources = 1;
    cp.initiator_depth = 1;
    cp.retry_count = 7;
    cp.rnr_retry_count = 7;  // retry on receiver-not-ready, don't error the QP
    if (rdma_connect(cid, &cp)) { Warn("rdma_connect", 0); return nullptr; }
    if (!await(RDMA_CM_EVENT_ESTABLISHED)) return nullptr;
    // The dial event channel is kept alive for the connection lifetime; we
    // don't process further events on it (RC stays up until Shutdown).
    return c;
  }
};

IbverbsTransport::IbverbsTransport(RdmaConfig cfg, OnReceiveFn on_receive)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = std::move(cfg);
  impl_->on_receive = std::move(on_receive);
  // gRPC fallback handles Fetch (read-repair) and gRPC-side Connect bookkeeping.
  impl_->grpc_fallback =
      std::make_unique<GrpcStreamTransport>(impl_->cfg, KvTransport::OnReceiveFn{});

  if (const char* l = std::getenv("LETHE_RDMA_LISTEN")) {
    std::string s(l);
    auto colon = s.rfind(':');
    if (colon != std::string::npos) {
      impl_->listen_ip = s.substr(0, colon);
      impl_->listen_port = static_cast<std::uint16_t>(std::stoi(s.substr(colon + 1)));
    }
  }
  impl_->peers = ParsePeerEnv(std::getenv("LETHE_RDMA_PEERS"));
  if (const char* m = std::getenv("LETHE_RDMA_MSG_BYTES")) {
    if (std::uint64_t v = std::strtoull(m, nullptr, 10)) impl_->msg_bytes = v;
  }
}

IbverbsTransport::~IbverbsTransport() { Shutdown(); }

void IbverbsTransport::Start() {
  impl_->grpc_fallback->Start();
  if (impl_->listen_ip.empty() || impl_->listen_port == 0) {
    std::fprintf(stderr, "[lethe][rdma] no LETHE_RDMA_LISTEN set; "
                         "RDMA receive side disabled\n");
    return;
  }
  impl_->listen_ch = rdma_create_event_channel();
  if (!impl_->listen_ch) { Warn("rdma_create_event_channel(listen)", 0); return; }
  if (rdma_create_id(impl_->listen_ch, &impl_->listen_id, nullptr, RDMA_PS_TCP)) {
    Warn("rdma_create_id(listen)", 0); return;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(impl_->listen_port);
  inet_pton(AF_INET, impl_->listen_ip.c_str(), &addr.sin_addr);
  if (rdma_bind_addr(impl_->listen_id, reinterpret_cast<sockaddr*>(&addr))) {
    Warn("rdma_bind_addr", 0); return;
  }
  if (rdma_listen(impl_->listen_id, 16)) { Warn("rdma_listen", 0); return; }
  std::fprintf(stderr, "[lethe][rdma] listening on %s:%u\n",
               impl_->listen_ip.c_str(), impl_->listen_port);
  impl_->listen_thread = std::thread([this] { impl_->ListenLoop(); });
}

void IbverbsTransport::Shutdown() {
  if (impl_->stop.exchange(true)) return;
  std::fprintf(stderr,
      "[lethe][rdma] stats: posted=%llu done_ok=%llu done_err=%llu "
      "drop_noconn=%llu drop_bp=%llu recv=%llu\n",
      (unsigned long long)impl_->n_posted.load(),
      (unsigned long long)impl_->n_done_ok.load(),
      (unsigned long long)impl_->n_done_err.load(),
      (unsigned long long)impl_->n_drop_noconn.load(),
      (unsigned long long)impl_->n_drop_bp.load(),
      (unsigned long long)impl_->n_recv.load());
  if (impl_->listen_id) rdma_destroy_id(impl_->listen_id);
  if (impl_->listen_ch) rdma_destroy_event_channel(impl_->listen_ch);
  if (impl_->listen_thread.joinable()) impl_->listen_thread.join();
  std::lock_guard<std::mutex> g(impl_->conns_mu);
  auto teardown = [](Conn* c) {
    c->stop.store(true);
    if (c->poller.joinable()) c->poller.join();
    if (c->id && c->id->qp) rdma_destroy_qp(c->id);
    for (auto& b : c->recv_bufs) if (b.mr) ibv_dereg_mr(b.mr);
    for (auto& b : c->send_bufs) if (b.mr) ibv_dereg_mr(b.mr);
    if (c->cq) ibv_destroy_cq(c->cq);
    if (c->pd) ibv_dealloc_pd(c->pd);
  };
  for (auto& [id, c] : impl_->outbound) teardown(c.get());
  for (auto& c : impl_->inbound) teardown(c.get());
  impl_->outbound.clear();
  impl_->inbound.clear();
  if (impl_->grpc_fallback) impl_->grpc_fallback->Shutdown();
}

void IbverbsTransport::Connect(const std::string& peer_id,
                               const std::string& address) {
  // gRPC side (for Fetch) connects now. The RDMA outbound QP is dialed
  // lazily on the first Send (see Impl::EnsureOutbound) so server startup
  // doesn't race peer listeners.
  impl_->grpc_fallback->Connect(peer_id, address);
  if (!impl_->peers.count(peer_id)) {
    std::fprintf(stderr, "[lethe][rdma] no RDMA endpoint for peer %s; "
                         "Send to it will fail (Fetch still works via gRPC)\n",
                 peer_id.c_str());
  }
}

void IbverbsTransport::Disconnect(const std::string& peer_id) {
  impl_->grpc_fallback->Disconnect(peer_id);
  std::lock_guard<std::mutex> g(impl_->conns_mu);
  auto it = impl_->outbound.find(peer_id);
  if (it == impl_->outbound.end()) return;
  Conn* c = it->second.get();
  c->stop.store(true);
  if (c->poller.joinable()) c->poller.join();
  if (c->id && c->id->qp) rdma_destroy_qp(c->id);
  for (auto& b : c->send_bufs) if (b.mr) ibv_dereg_mr(b.mr);
  if (c->cq) ibv_destroy_cq(c->cq);
  if (c->pd) ibv_dealloc_pd(c->pd);
  impl_->outbound.erase(it);
}

std::future<bool> IbverbsTransport::Send(const std::string& peer_id,
                                         BlockId id,
                                         StreamPurpose purpose,
                                         std::span<const std::byte> data) {
  auto pr = std::make_shared<std::promise<bool>>();
  auto fut = pr->get_future();

  if (kHeaderBytes + data.size() > impl_->MsgCap()) {
    pr->set_value(false);   // block bigger than the registered buffer
    return fut;
  }

  Conn* c = impl_->EnsureOutbound(peer_id);  // lazy dial on first Send
  if (!c) { impl_->n_drop_noconn.fetch_add(1); pr->set_value(false); return fut; }

  // Acquire a free send buffer. If the ring is full, wait (bounded) for a
  // completion to free one rather than dropping instantly — the re-replication
  // bounce roughly doubles send volume, so transient bursts exceed the ring.
  // Never hold send_mu while sleeping (PollOutbound needs it to free buffers).
  int buf_idx = -1;
  std::uint64_t wr_id = 0;
  for (int attempt = 0; attempt < 2000 && buf_idx < 0; ++attempt) {
    {
      std::lock_guard<std::mutex> g(c->send_mu);
      if (!c->free_send.empty()) {
        buf_idx = c->free_send.back();
        c->free_send.pop_back();
        wr_id = (c->next_wr++ << 16) | static_cast<std::uint64_t>(buf_idx & 0xffff);
        c->pending[wr_id] = pr;
      }
    }
    if (buf_idx < 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  if (buf_idx < 0) { impl_->n_drop_bp.fetch_add(1); pr->set_value(false); return fut; }

  RegBuf& b = c->send_bufs[buf_idx];
  auto* h = reinterpret_cast<MsgHeader*>(b.mem.data());
  h->magic = kMagic;
  h->type = kMsgBlockPush;
  h->purpose = static_cast<std::uint32_t>(purpose);
  h->payload_len = static_cast<std::uint32_t>(data.size());
  std::memcpy(h->hash, id.hash.data(), 32);
  h->layer = id.layer;
  h->head_group = id.head_group;
  h->model_id = id.model_id;
  std::memcpy(b.mem.data() + kHeaderBytes, data.data(), data.size());

  ibv_sge sge{};
  sge.addr = reinterpret_cast<std::uintptr_t>(b.mem.data());
  sge.length = static_cast<std::uint32_t>(kHeaderBytes + data.size());
  sge.lkey = b.mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id = wr_id;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  ibv_send_wr* bad = nullptr;
  int rc = ibv_post_send(c->id->qp, &wr, &bad);
  if (rc) {
    Warn("ibv_post_send", rc);
    std::lock_guard<std::mutex> g(c->send_mu);
    c->pending.erase(wr_id);
    c->free_send.push_back(buf_idx);
    pr->set_value(false);
  } else {
    impl_->n_posted.fetch_add(1);
  }
  return fut;
}

std::future<std::optional<KvBlock>> IbverbsTransport::Fetch(
    const std::string& peer_id, BlockId id) {
  // Read-repair pull goes over gRPC (see file header).
  return impl_->grpc_fallback->Fetch(peer_id, id);
}

}  // namespace lethe
