// Lethe — IbverbsTransport (compiled only when LETHE_ENABLE_RDMA=ON).
//
// W5-6 STATE: this implementation is a header-symbol stub. GATE #1 of
// the W5-6 milestone fired — the WSL2 dev kernel
// (5.15.x-microsoft-standard-WSL2) ships without the InfiniBand
// subsystem, so SoftRoCE could not be brought up to validate the
// ibverbs codepath. Rather than write speculative RDMA code that no
// CI machine can exercise, the methods below provide DEFINED symbols
// (so cmake -DLETHE_ENABLE_RDMA=ON still links cleanly) but fail
// loudly if anyone actually constructs the class at runtime. See
// docs/decisions/W5_rdma_fallback.md for the rationale.
//
// When real IB hardware (or a SoftRoCE-capable kernel) is available at
// W12 PACE testing, this file gets the real implementation:
//   - rdma_cm-based connection establishment
//   - per-peer RC QP, pinned send/recv buffer pool
//   - dedicated completion-polling thread per peer (per CLAUDE.md
//     "Threading invariants")
//   - Send → ibv_post_send on the send queue
//   - Fetch → either request-via-Send + response-via-OnReceive, or
//     RDMA Read against a peer's exported MR (decision deferred)
//
// The KvTransport abstraction (kv_transport.hpp) is intentionally
// shaped so that swap is local to this file plus the factory in
// main.cpp.

#include "lethe/kv_transport.hpp"

#include <cstdio>
#include <cstdlib>
#include <future>
#include <optional>
#include <utility>

#include "lethe/types.hpp"

namespace lethe {

namespace {

[[noreturn]] void UnimplementedAbort(const char* method) {
  std::fprintf(stderr,
               "[lethe] IbverbsTransport::%s called but RDMA is not "
               "implemented for this build. See "
               "docs/decisions/W5_rdma_fallback.md.\n",
               method);
  std::abort();
}

}  // namespace

struct IbverbsTransport::Impl {
  RdmaConfig cfg;
  OnReceiveFn on_receive;
};

IbverbsTransport::IbverbsTransport(RdmaConfig cfg, OnReceiveFn on_receive)
    : impl_(std::make_unique<Impl>()) {
  impl_->cfg = std::move(cfg);
  impl_->on_receive = std::move(on_receive);
}

IbverbsTransport::~IbverbsTransport() = default;

void IbverbsTransport::Start() { UnimplementedAbort("Start"); }
void IbverbsTransport::Shutdown() { UnimplementedAbort("Shutdown"); }

void IbverbsTransport::Connect(const std::string& /*peer_id*/,
                               const std::string& /*address*/) {
  UnimplementedAbort("Connect");
}

void IbverbsTransport::Disconnect(const std::string& /*peer_id*/) {
  UnimplementedAbort("Disconnect");
}

std::future<bool> IbverbsTransport::Send(const std::string& /*peer_id*/,
                                         BlockId /*id*/,
                                         StreamPurpose /*purpose*/,
                                         std::span<const std::byte> /*data*/) {
  UnimplementedAbort("Send");
}

std::future<std::optional<KvBlock>> IbverbsTransport::Fetch(
    const std::string& /*peer_id*/, BlockId /*id*/) {
  UnimplementedAbort("Fetch");
}

}  // namespace lethe
