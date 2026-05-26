// Metrics unit test.
//
// Standalone main()-with-asserts. Drives the Metrics class directly via
// the Record* methods and scrape_for_testing() (the same exposition the
// /metrics HTTP handler serves), asserting:
//   - Record* update the right metric family / labels.
//   - The exposition is well-formed Prometheus text (HELP/TYPE lines,
//     name{labels} value).
//   - Histogram buckets accumulate cumulatively and _count/_sum track.
//   - Thread-safe under concurrent Record* (the real value is under
//     TSan; functionally we just assert the totals add up).
//
// The Metrics ctor binds an ephemeral /metrics port (0) - the HTTP
// thread runs but the test never connects to it; it scrapes in-process.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "lethe/metrics.hpp"
#include "lethe/types.hpp"

using namespace lethe;
using namespace std::chrono;

namespace {

// Does `haystack` contain `needle`?
bool contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// Extract the integer value following a metric line that starts with
// `prefix` (the full "name{labels}" up to the value). Returns -1 if not
// found.
long long value_after(const std::string& text, const std::string& prefix) {
  auto pos = text.find(prefix);
  if (pos == std::string::npos) return -1;
  pos += prefix.size();
  // skip spaces
  while (pos < text.size() && text[pos] == ' ') ++pos;
  long long v = 0;
  bool any = false;
  while (pos < text.size() && (isdigit(text[pos]) || text[pos] == '-')) {
    v = v * 10 + (text[pos] - '0');
    ++pos;
    any = true;
  }
  return any ? v : -1;
}

Metrics make_metrics() {
  // Ephemeral port; HTTP thread harmless for the unit test.
  return Metrics("0.0.0.0:0", "testnode");
}

void TestRecordLookupHitMiss() {
  Metrics m = make_metrics();
  m.RecordLookup(3, 1, nanoseconds(120000));   // 3 hits, 1 miss, 120us
  m.RecordLookup(2, 2, nanoseconds(800000));   // 2 hits, 2 miss, 800us
  std::string s = m.scrape_for_testing();

  // Families present.
  assert(contains(s, "# TYPE lethe_requests_total counter"));
  assert(contains(s, "# TYPE lethe_op_latency_seconds histogram"));

  // hit = 5, miss = 3.
  long long hit = value_after(
      s, "lethe_requests_total{node=\"testnode\",op=\"lookup\",result=\"hit\"}");
  long long miss = value_after(
      s, "lethe_requests_total{node=\"testnode\",op=\"lookup\",result=\"miss\"}");
  assert(hit == 5);
  assert(miss == 3);

  // Latency histogram count == 2 observations.
  long long count = value_after(
      s, "lethe_op_latency_seconds_count{node=\"testnode\",op=\"lookup\"}");
  assert(count == 2);
  std::printf("  TestRecordLookupHitMiss: ok (hit=%lld miss=%lld)\n", hit, miss);
}

void TestHistogramBucketsCumulative() {
  Metrics m = make_metrics();
  // Observations: 30us (bucket le=5e-5), 200us (le=5e-4), 2ms (le=5e-3),
  // 200ms (+Inf).
  m.RecordLookup(1, 0, nanoseconds(30000));    // 30us
  m.RecordLookup(1, 0, nanoseconds(200000));   // 200us
  m.RecordLookup(1, 0, nanoseconds(2000000));  // 2ms
  m.RecordLookup(1, 0, nanoseconds(200000000));// 200ms → +Inf only
  std::string s = m.scrape_for_testing();

  // Cumulative buckets: le=5e-5 has 1 (the 30us); le=0.0005 has 2
  // (30us + 200us); le=+Inf has all 4.
  long long le_50us = value_after(
      s, "lethe_op_latency_seconds_bucket{node=\"testnode\",op=\"lookup\",le=\"5e-05\"}");
  long long le_inf = value_after(
      s, "lethe_op_latency_seconds_bucket{node=\"testnode\",op=\"lookup\",le=\"+Inf\"}");
  long long count = value_after(
      s, "lethe_op_latency_seconds_count{node=\"testnode\",op=\"lookup\"}");
  assert(le_50us == 1);
  assert(le_inf == 4);
  assert(count == 4);
  // Cumulative monotonicity: le=+Inf >= le=5e-05.
  assert(le_inf >= le_50us);
  std::printf("  TestHistogramBucketsCumulative: ok (le50us=%lld leInf=%lld)\n",
              le_50us, le_inf);
}

void TestGaugesAndCounters() {
  Metrics m = make_metrics();
  m.RecordEpoch(7);
  m.RecordTierUsage(Tier::DRAM, 4096, 1 << 20);
  m.RecordStreamBytes(8192, nanoseconds(500000));
  m.RecordEvictionPass(5, 5000);
  m.RecordUnderReplicated(3);
  std::string s = m.scrape_for_testing();

  assert(value_after(s, "lethe_cluster_epoch{node=\"testnode\"}") == 7);
  assert(value_after(
      s, "lethe_tier_bytes{node=\"testnode\",tier=\"dram\",state=\"used\"}") == 4096);
  assert(value_after(s, "lethe_stream_bytes_total{node=\"testnode\"}") == 8192);
  assert(value_after(s, "lethe_eviction_blocks_total{node=\"testnode\"}") == 5);
  assert(value_after(s, "lethe_replicas_under_target{node=\"testnode\"}") == 3);
  // TYPE lines present for the dashboard's families.
  assert(contains(s, "# TYPE lethe_tier_bytes gauge"));
  assert(contains(s, "# TYPE lethe_cluster_epoch gauge"));
  assert(contains(s, "# TYPE lethe_stream_bytes_total counter"));
  std::printf("  TestGaugesAndCounters: ok\n");
}

void TestConcurrentRecord() {
  Metrics m = make_metrics();
  constexpr int kThreads = 8;
  constexpr int kPerThread = 10000;
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) {
    ts.emplace_back([&m]() {
      for (int i = 0; i < kPerThread; ++i) {
        m.RecordLookup(1, 1, nanoseconds(100000));
      }
    });
  }
  for (auto& th : ts) th.join();
  std::string s = m.scrape_for_testing();
  long long hit = value_after(
      s, "lethe_requests_total{node=\"testnode\",op=\"lookup\",result=\"hit\"}");
  long long miss = value_after(
      s, "lethe_requests_total{node=\"testnode\",op=\"lookup\",result=\"miss\"}");
  const long long expected = static_cast<long long>(kThreads) * kPerThread;
  assert(hit == expected);
  assert(miss == expected);
  std::printf("  TestConcurrentRecord: ok (hit=%lld == %lld)\n", hit, expected);
}

}  // namespace

int main() {
  std::printf("test_metrics:\n");
  TestRecordLookupHitMiss();
  TestHistogramBucketsCumulative();
  TestGaugesAndCounters();
  TestConcurrentRecord();
  std::printf("test_metrics: ALL PASS\n");
  return 0;
}
