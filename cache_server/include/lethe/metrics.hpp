#pragma once
// Lethe — Prometheus metrics (W10).
//
// Uses prometheus-cpp. Metrics fall into four families:
//
//   - lethe_requests_total{op="lookup|insert|stream",result="hit|miss|err"}
//   - lethe_op_latency_seconds (histogram, per op)
//   - lethe_tier_bytes{tier="hbm|dram|ssd",state="used|capacity"}
//   - lethe_cluster_epoch (gauge)
//
// Plus a small surface for the chaos harness to assert on
// (lethe_failover_recovery_seconds histogram, lethe_replicas_under_target counter).

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include "lethe/types.hpp"

namespace lethe {

class Metrics {
 public:
  Metrics(const std::string& bind_address, const std::string& node_id);
  ~Metrics();

  void RecordLookup(std::size_t hits, std::size_t misses,
                    std::chrono::nanoseconds latency);
  void RecordInsert(std::size_t accepted, std::chrono::nanoseconds latency);
  void RecordStreamBytes(std::uint64_t bytes,
                         std::chrono::nanoseconds latency);
  void RecordTierUsage(Tier tier, std::size_t used_bytes, std::size_t capacity);
  void RecordEpoch(std::uint64_t epoch);
  void RecordEvictionPass(std::size_t blocks_evicted, std::size_t bytes_freed);
  void RecordFailoverRecovery(std::chrono::milliseconds duration);
  void RecordUnderReplicated(std::size_t count);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Scoped helper for measuring op latency.
class LatencyTimer {
 public:
  LatencyTimer() : start_(std::chrono::steady_clock::now()) {}
  std::chrono::nanoseconds elapsed() const {
    return std::chrono::steady_clock::now() - start_;
  }
 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace lethe
