// Lethe — metrics implementation (W10).
//
// W1 ships only the destructor stub so LetheCache's unique_ptr<Metrics>
// is destructible. Real prometheus-cpp hookup lands in W10; until
// then Metrics is pimpl-shaped with an empty Impl so the destructor
// has a complete type to delete.

#include "lethe/metrics.hpp"

namespace lethe {

struct Metrics::Impl {};

Metrics::Metrics(const std::string& /*bind_address*/,
                 const std::string& /*node_id*/)
    : impl_(std::make_unique<Impl>()) {}

Metrics::~Metrics() = default;

// W10: RecordLookup, RecordInsert, RecordStreamBytes, RecordTierUsage,
// RecordEpoch, RecordEvictionPass, RecordFailoverRecovery,
// RecordUnderReplicated all become real prometheus-cpp histogram /
// gauge / counter updates.
void Metrics::RecordLookup(std::size_t, std::size_t,
                           std::chrono::nanoseconds) {}
void Metrics::RecordInsert(std::size_t, std::chrono::nanoseconds) {}
void Metrics::RecordStreamBytes(std::uint64_t, std::chrono::nanoseconds) {}
void Metrics::RecordTierUsage(Tier, std::size_t, std::size_t) {}
void Metrics::RecordEpoch(std::uint64_t) {}
void Metrics::RecordEvictionPass(std::size_t, std::size_t) {}
void Metrics::RecordFailoverRecovery(std::chrono::milliseconds) {}
void Metrics::RecordUnderReplicated(std::size_t) {}

}  // namespace lethe
