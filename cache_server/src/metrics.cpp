// Metrics - hand-rolled Prometheus text-exposition exporter. prometheus-cpp
// isn't packaged for Ubuntu 22.04 and building it from source pulls in
// civetweb + a protobuf metrics path we don't need; our metric families are
// simple and metrics.hpp is pimpl'd so the backend is an implementation
// detail.
//
// Storage / synchronization: ALL metric state is std::atomic, so every Record*
// call is lock-free and sub-microsecond - no mutex on the hot Lookup/Insert
// path. The series are fixed and pre-created at construction, so Record*
// updates a known member directly with no map lookup. The /metrics renderer
// reads the atomics (relaxed); a scrape may see a momentarily-inconsistent
// snapshot across series, which is fine for metrics.
//
// HTTP: a single dedicated thread runs a minimal raw-POSIX-socket server. It
// answers any GET path with the exposition; no routing, no keep-alive.
// SO_RCVTIMEO on the listen socket makes accept() wake periodically so
// Shutdown is responsive. Bind failure (e.g. a port taken by a sibling node in
// a shared-host test) is non-fatal: we log and run without a metrics endpoint
// rather than crashing.

#include "lethe/metrics.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace lethe {

namespace {

// Latency histogram bounds in SECONDS (microsecond-to-millisecond ops
// on loopback gRPC). 8 finite bounds → 9 buckets (last is +Inf).
constexpr std::array<double, 8> kLatencyBounds = {
    5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1,
};
// Failover recovery is a seconds-scale event (3.5s budget); coarser
// bounds spanning ~0.5s..8s.
constexpr std::array<double, 8> kRecoveryBounds = {
    0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 8.0,
};

struct Histogram {
  const std::array<double, 8>& bounds;
  std::array<std::atomic<std::uint64_t>, 9> buckets{};  // per-bucket counts
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> sum_micros{0};  // integer accumulation

  explicit Histogram(const std::array<double, 8>& b) : bounds(b) {}

  void observe_seconds(double sec) {
    std::size_t i = 0;
    while (i < bounds.size() && sec > bounds[i]) ++i;
    buckets[i].fetch_add(1, std::memory_order_relaxed);
    count.fetch_add(1, std::memory_order_relaxed);
    sum_micros.fetch_add(static_cast<std::uint64_t>(sec * 1e6),
                         std::memory_order_relaxed);
  }
};

void render_counter(std::ostream& os, const std::string& name,
                    const std::string& help, const std::string& labels,
                    std::uint64_t v) {
  os << "# HELP " << name << " " << help << "\n";
  os << "# TYPE " << name << " counter\n";
  os << name << labels << " " << v << "\n";
}

void render_gauge(std::ostream& os, const std::string& name,
                  const std::string& help, const std::string& labels,
                  long long v) {
  os << "# HELP " << name << " " << help << "\n";
  os << "# TYPE " << name << " gauge\n";
  os << name << labels << " " << v << "\n";
}

void render_histogram(std::ostream& os, const std::string& family,
                      const std::string& op_label, Histogram& h) {
  // Cumulative buckets per the Prometheus histogram contract.
  std::uint64_t cum = 0;
  for (std::size_t i = 0; i < h.bounds.size(); ++i) {
    cum += h.buckets[i].load(std::memory_order_relaxed);
    os << family << "_bucket{" << op_label
       << (op_label.empty() ? "" : ",") << "le=\"" << h.bounds[i] << "\"} "
       << cum << "\n";
  }
  cum += h.buckets[8].load(std::memory_order_relaxed);
  os << family << "_bucket{" << op_label
     << (op_label.empty() ? "" : ",") << "le=\"+Inf\"} " << cum << "\n";
  const double sum_sec =
      h.sum_micros.load(std::memory_order_relaxed) / 1e6;
  os << family << "_sum{" << op_label << "} " << sum_sec << "\n";
  os << family << "_count{" << op_label << "} "
     << h.count.load(std::memory_order_relaxed) << "\n";
}

int tier_index(Tier t) {
  switch (t) {
    case Tier::HBM:  return 0;
    case Tier::DRAM: return 1;
    case Tier::SSD:  return 2;
  }
  return 1;
}
const char* tier_name(int i) {
  return i == 0 ? "hbm" : (i == 1 ? "dram" : "ssd");
}

}  // namespace

struct Metrics::Impl {
  std::string node_id;
  std::string bind_address;

  // --- counters ---
  std::atomic<std::uint64_t> req_lookup_hit{0};
  std::atomic<std::uint64_t> req_lookup_miss{0};
  std::atomic<std::uint64_t> req_insert_ok{0};
  std::atomic<std::uint64_t> req_stream_ok{0};
  std::atomic<std::uint64_t> stream_bytes_total{0};
  std::atomic<std::uint64_t> eviction_blocks_total{0};
  std::atomic<std::uint64_t> eviction_bytes_freed_total{0};

  // --- gauges ---
  std::atomic<long long> cluster_epoch{0};
  std::atomic<long long> tier_used[3]{};
  std::atomic<long long> tier_capacity[3]{};
  std::atomic<long long> replicas_under_target{0};

  // --- histograms ---
  Histogram lat_lookup{kLatencyBounds};
  Histogram lat_insert{kLatencyBounds};
  Histogram lat_stream{kLatencyBounds};
  Histogram failover_recovery{kRecoveryBounds};

  // --- HTTP server ---
  std::atomic<bool> running{false};
  int listen_fd = -1;
  std::thread http_thread;

  std::string render() {
    std::ostringstream os;
    const std::string inst = "node=\"" + node_id + "\"";

    // lethe_requests_total{op,result}
    os << "# HELP lethe_requests_total Cache requests by op and result\n";
    os << "# TYPE lethe_requests_total counter\n";
    os << "lethe_requests_total{" << inst << ",op=\"lookup\",result=\"hit\"} "
       << req_lookup_hit.load(std::memory_order_relaxed) << "\n";
    os << "lethe_requests_total{" << inst << ",op=\"lookup\",result=\"miss\"} "
       << req_lookup_miss.load(std::memory_order_relaxed) << "\n";
    os << "lethe_requests_total{" << inst << ",op=\"insert\",result=\"ok\"} "
       << req_insert_ok.load(std::memory_order_relaxed) << "\n";
    os << "lethe_requests_total{" << inst << ",op=\"stream\",result=\"ok\"} "
       << req_stream_ok.load(std::memory_order_relaxed) << "\n";

    // lethe_op_latency_seconds (histogram, per op)
    os << "# HELP lethe_op_latency_seconds Operation latency seconds\n";
    os << "# TYPE lethe_op_latency_seconds histogram\n";
    render_histogram(os, "lethe_op_latency_seconds",
                     inst + ",op=\"lookup\"", lat_lookup);
    render_histogram(os, "lethe_op_latency_seconds",
                     inst + ",op=\"insert\"", lat_insert);
    render_histogram(os, "lethe_op_latency_seconds",
                     inst + ",op=\"stream\"", lat_stream);

    // lethe_tier_bytes{tier,state}
    os << "# HELP lethe_tier_bytes Tier byte usage and capacity\n";
    os << "# TYPE lethe_tier_bytes gauge\n";
    for (int i = 0; i < 3; ++i) {
      os << "lethe_tier_bytes{" << inst << ",tier=\"" << tier_name(i)
         << "\",state=\"used\"} "
         << tier_used[i].load(std::memory_order_relaxed) << "\n";
      os << "lethe_tier_bytes{" << inst << ",tier=\"" << tier_name(i)
         << "\",state=\"capacity\"} "
         << tier_capacity[i].load(std::memory_order_relaxed) << "\n";
    }

    render_gauge(os, "lethe_cluster_epoch", "Cluster membership epoch",
                 "{" + inst + "}",
                 cluster_epoch.load(std::memory_order_relaxed));
    render_gauge(os, "lethe_replicas_under_target",
                 "Blocks currently below replication target",
                 "{" + inst + "}",
                 replicas_under_target.load(std::memory_order_relaxed));

    render_counter(os, "lethe_stream_bytes_total",
                   "Total bytes transferred via the KV transport",
                   "{" + inst + "}",
                   stream_bytes_total.load(std::memory_order_relaxed));
    os << "# HELP lethe_eviction_blocks_total Blocks evicted\n";
    os << "# TYPE lethe_eviction_blocks_total counter\n";
    os << "lethe_eviction_blocks_total{" << inst << "} "
       << eviction_blocks_total.load(std::memory_order_relaxed) << "\n";
    os << "# HELP lethe_eviction_bytes_freed_total Bytes freed by eviction\n";
    os << "# TYPE lethe_eviction_bytes_freed_total counter\n";
    os << "lethe_eviction_bytes_freed_total{" << inst << "} "
       << eviction_bytes_freed_total.load(std::memory_order_relaxed) << "\n";

    // lethe_failover_recovery_seconds (histogram)
    os << "# HELP lethe_failover_recovery_seconds Failover recovery seconds\n";
    os << "# TYPE lethe_failover_recovery_seconds histogram\n";
    render_histogram(os, "lethe_failover_recovery_seconds", inst,
                     failover_recovery);

    return os.str();
  }

  void serve(std::uint16_t port) {
    listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
      std::cerr << "[lethe] metrics: socket() failed; no /metrics endpoint\n";
      return;
    }
    int one = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    // Wake accept() periodically so Shutdown is responsive.
    timeval tv{};
    tv.tv_sec = 0;
    tv.tv_usec = 400000;  // 400ms
    ::setsockopt(listen_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      std::cerr << "[lethe] metrics: bind(0.0.0.0:" << port
                << ") failed (" << std::strerror(errno)
                << "); node runs without a /metrics endpoint\n";
      ::close(listen_fd);
      listen_fd = -1;
      return;
    }
    if (::listen(listen_fd, 16) != 0) {
      std::cerr << "[lethe] metrics: listen() failed; no /metrics endpoint\n";
      ::close(listen_fd);
      listen_fd = -1;
      return;
    }
    std::cout << "[lethe] metrics: /metrics on 0.0.0.0:" << port << "\n";

    while (running.load(std::memory_order_acquire)) {
      int c = ::accept(listen_fd, nullptr, nullptr);
      if (c < 0) continue;  // timeout (EAGAIN) or interrupted → re-check running
      // Drain the request (we don't parse it - any GET gets the exposition).
      char buf[2048];
      ::recv(c, buf, sizeof(buf), 0);
      const std::string body = render();
      std::ostringstream resp;
      resp << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: text/plain; version=0.0.4\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n\r\n"
           << body;
      const std::string out = resp.str();
      ::send(c, out.data(), out.size(), 0);
      ::close(c);
    }
  }
};

Metrics::Metrics(const std::string& bind_address, const std::string& node_id)
    : impl_(std::make_unique<Impl>()) {
  impl_->node_id = node_id;
  impl_->bind_address = bind_address;

  // Parse "host:port" → port (we always bind 0.0.0.0). Default 9090.
  std::uint16_t port = 9090;
  auto colon = bind_address.rfind(':');
  if (colon != std::string::npos) {
    try {
      port = static_cast<std::uint16_t>(std::stoi(bind_address.substr(colon + 1)));
    } catch (...) {
      port = 9090;
    }
  }

  impl_->running.store(true, std::memory_order_release);
  Impl* impl = impl_.get();
  impl_->http_thread = std::thread([impl, port]() { impl->serve(port); });
}

Metrics::~Metrics() {
  impl_->running.store(false, std::memory_order_release);
  if (impl_->http_thread.joinable()) impl_->http_thread.join();
  if (impl_->listen_fd >= 0) ::close(impl_->listen_fd);
}

void Metrics::RecordLookup(std::size_t hits, std::size_t misses,
                           std::chrono::nanoseconds latency) {
  impl_->req_lookup_hit.fetch_add(hits, std::memory_order_relaxed);
  impl_->req_lookup_miss.fetch_add(misses, std::memory_order_relaxed);
  impl_->lat_lookup.observe_seconds(
      std::chrono::duration<double>(latency).count());
}

void Metrics::RecordInsert(std::size_t accepted,
                           std::chrono::nanoseconds latency) {
  impl_->req_insert_ok.fetch_add(accepted, std::memory_order_relaxed);
  impl_->lat_insert.observe_seconds(
      std::chrono::duration<double>(latency).count());
}

void Metrics::RecordStreamBytes(std::uint64_t bytes,
                                std::chrono::nanoseconds latency) {
  impl_->req_stream_ok.fetch_add(1, std::memory_order_relaxed);
  impl_->stream_bytes_total.fetch_add(bytes, std::memory_order_relaxed);
  impl_->lat_stream.observe_seconds(
      std::chrono::duration<double>(latency).count());
}

void Metrics::RecordTierUsage(Tier tier, std::size_t used_bytes,
                              std::size_t capacity) {
  const int i = tier_index(tier);
  impl_->tier_used[i].store(static_cast<long long>(used_bytes),
                            std::memory_order_relaxed);
  impl_->tier_capacity[i].store(static_cast<long long>(capacity),
                                std::memory_order_relaxed);
}

void Metrics::RecordEpoch(std::uint64_t epoch) {
  impl_->cluster_epoch.store(static_cast<long long>(epoch),
                             std::memory_order_relaxed);
}

void Metrics::RecordEvictionPass(std::size_t blocks_evicted,
                                 std::size_t bytes_freed) {
  impl_->eviction_blocks_total.fetch_add(blocks_evicted,
                                         std::memory_order_relaxed);
  impl_->eviction_bytes_freed_total.fetch_add(bytes_freed,
                                              std::memory_order_relaxed);
}

void Metrics::RecordFailoverRecovery(std::chrono::milliseconds duration) {
  impl_->failover_recovery.observe_seconds(
      std::chrono::duration<double>(duration).count());
}

void Metrics::RecordUnderReplicated(std::size_t count) {
  impl_->replicas_under_target.store(static_cast<long long>(count),
                                     std::memory_order_relaxed);
}

std::string Metrics::scrape_for_testing() const {
  return impl_->render();
}

}  // namespace lethe
