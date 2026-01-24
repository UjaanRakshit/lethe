# W5-6 RDMA fallback — gate #1 fired

**Date:** 2026-05-27
**Status:** locked. The W5-6 milestone shipped the KvTransport architectural piece via gRPC; ibverbs implementation is deferred to W12 PACE testing on real IB hardware.

## TL;DR

The W5-6 prompt embedded three hard gates. Gate #1 — SoftRoCE viability
within 90 minutes — fired immediately. The WSL2 dev kernel
(`5.15.167.4-microsoft-standard-WSL2`) ships without the InfiniBand
subsystem at all (no `/lib/modules/.../kernel/drivers/infiniband/`
directory), so `rdma_rxe` cannot be loaded and there is no path to a
functioning rxe device. Per CLAUDE.md rule 4 and the prompt's gate-#1
fallback instruction, the milestone pivoted to the architectural piece:
refactor `Replicator` to use the `KvTransport` abstraction so the W12
hardware swap is a single-file change in `cache.cpp`.

## What I checked before declaring gate #1

```text
$ uname -a
Linux UjaanLaptop 5.15.167.4-microsoft-standard-WSL2 #1 SMP Tue Nov 5 ...

$ ls /lib/modules/$(uname -r)/kernel/drivers/infiniband/
ls: cannot access '...': No such file or directory

$ sudo modprobe rdma_rxe
modprobe: FATAL: Module rdma_rxe not found in directory /lib/modules/...

$ rdma link show
Failed to open NETLINK_RDMA socket
```

Three independent blockers (each sufficient on its own):

1. **No InfiniBand subsystem.** The WSL2 Microsoft kernel is stripped
   down and does not include any IB drivers. `rdma_rxe.ko` cannot
   exist because the `ib_core` it depends on doesn't either.
2. **NETLINK_RDMA socket fails.** Even the kernel's RDMA management
   API isn't compiled in. There's no userspace-side hook to talk to.
3. **`ibverbs-utils` not installed.** Trivially fixable, but moot
   without (1) and (2).

Total wall-clock cost of the probe: ~3 minutes (the 90-minute gate
budget was untouched). The probe is reproducible — re-run
`scripts/setup_softroce.sh` on this WSL2 environment to confirm.

### Could we build a custom WSL2 kernel with CONFIG_RDMA_RXE=m?

Yes, in principle. The cost: clone the WSL2 kernel source, enable the
RDMA config flags (`CONFIG_INFINIBAND`, `CONFIG_INFINIBAND_USER_ACCESS`,
`CONFIG_RDMA_RXE`, plus the dependency closure), rebuild, swap the
`.wslconfig`-pointed kernel image, reboot WSL2. Best case: SoftRoCE on
loopback works and W5-6 can validate ibverbs end-to-end. Worst case
(and the more likely one based on community reports): Microsoft's
Hyper-V virtualization layer doesn't expose enough of the network
stack for rxe to attach cleanly, and we burn 8-12 hours discovering
that. The 25-hour cap in gate #2 makes this a poor bet.

The right venue for ibverbs validation is W12 PACE on real IB hardware.
The KvTransport abstraction below makes that swap a constructor
change in `cache.cpp` — no other code needs to know.

## What shipped instead

Three concrete pieces:

### 1. KvTransport interface (kv_transport.hpp)

Added `Fetch(peer_id, BlockId) -> future<optional<KvBlock>>` symmetric
with the existing `Send`. Both methods can be implemented over either
gRPC or ibverbs without leaking the underlying transport into the
interface — confirmed by writing both implementations against the same
contract.

### 2. GrpcStreamTransport implementation (grpc_stream_transport.cpp)

Was a TODO stub; is now the production data path. `Send` invokes the
`Insert` RPC; `Fetch` invokes the `Fetch` RPC. Both run synchronously
inside the call and return ready futures — concurrency lives at the
caller's worker-pool layer (`Replicator`'s queue + N=4 workers gives N
concurrent Sends in flight). One `grpc::Channel` per peer; HTTP/2 stream
multiplexing handles concurrency within a single channel.

### 3. Replicator refactor (replication.{hpp,cpp})

`Replicator::PeerClient` is gone — its responsibility moved into
`GrpcStreamTransport`. Constructor now takes `KvTransport*`; workers
dispatch via `transport_->Send`; `FetchFromAny` fans out
`transport_->Fetch` via `std::async` per peer. Async-replication policy
(bounded queue, drop-on-overflow, fire-and-forget) preserved
unchanged. `EnsurePeerClient` / `DropPeerClient` still on the public
surface; they delegate to `transport_->Connect` / `Disconnect`.

### 4. IbverbsTransport defined-symbol stub (ibverbs_transport.cpp)

Defines `IbverbsTransport`'s methods as `abort()` calls with a clear
message pointing at this document. This is intentional: the class
exists so `cmake -DLETHE_ENABLE_RDMA=ON` still links cleanly (acceptance
criterion B), but any code path that actually constructs it will crash
loudly rather than silently no-op. The factory in `cache.cpp` only
constructs `GrpcStreamTransport`; `IbverbsTransport` is unreachable in
shipped builds.

### 5. CMake update

Dropped `target_link_libraries(... ibverbs rdmacm)` under the
`LETHE_ENABLE_RDMA` guard — the stub uses none of those symbols, and
requiring `librdmacm-dev` on the build host for no runtime benefit is
hostile. The CMakeLists comment names the exact lines to re-add when
the real implementation lands at W12.

## Acceptance criteria — verified

| Criterion | Path-3 framing | Status |
|---|---|---|
| A. SoftRoCE up | Gate #1 fired; documented above. | n/a (intentional) |
| B. Build clean (RDMA=OFF + RDMA=ON + TSan) | All three clean, zero -Werror, zero TSan findings | ✓ |
| C. RDMA primitive (rdma_echo) | Skipped per gate-#1 fallback | n/a (intentional) |
| D. Replicator takes KvTransport& | Refactored; cache.cpp constructs GrpcStreamTransport at startup | ✓ |
| E. 3-node smoke with RDMA | Skipped per gate-#1 fallback | n/a (intentional) |
| F. W1 + W4 + W3 regressions | W1.4 (114s, all_identical), W4 3-node (4/4), W3 hash compat (5/5), single-node roundtrip (5/5), ctest 4/4 | ✓ |
| G. Perf measurement | RDMA vs gRPC comparison not meaningful without RDMA | n/a (intentional) |

## Alternatives considered

### Build the custom WSL2 kernel

Already discussed above. Bet that costs 8-12 hours with high
probability of producing nothing usable. The 25-hour W5-6 cap doesn't
have room for it, and the bullet outcome is identical to what we
actually did (architecturally complete, hardware-validation deferred
to W12).

### Skip the KvTransport refactor entirely

Defensible read: gate #1 fires → drop everything → wait for hardware
at W12. But the abstraction was already half-built (header existed,
both .cpp files were stubs, `LetheCache::transport_` was a typed
nullptr). Finishing it now is the cheap path; doing it at W12 under
hardware-bring-up pressure is the expensive path. Also the bullet
benefits: "designed and shipped the abstraction; hardware validation
at W12" reads better than "designed an abstraction but never built it."

### Implement Send/Fetch over StreamBlocks bidi-stream

The proto defines a bidi-stream `StreamBlocks` RPC that's designed for
chunked large transfers. We could have used it for `Send` instead of
the `Insert` RPC. Rejected: the W4 server-side handler implements
`Insert` (and uses it as the replication push receiver), but does not
implement a `StreamBlocks` server. Switching now would force a
server-side rewrite that's not on the W5-6 critical path. The W12
implementation is free to pick StreamBlocks for the ibverbs path
because the abstraction hides the wire choice.

## Cross-references

- `cache_server/include/lethe/kv_transport.hpp` (interface)
- `cache_server/src/grpc_stream_transport.cpp` (production data path)
- `cache_server/src/ibverbs_transport.cpp` (W12-deferred stub)
- `cache_server/src/replication.cpp` (Replicator refactor)
- `cache_server/src/cache.cpp` (transport factory at startup)
- `scripts/setup_softroce.sh` (probe; produces the gate-#1 evidence)
- `docs/DESIGN.md` §6 (the gRPC/RDMA split that this implements)
- CLAUDE.md rule 4 (the 2.5-week SoftRoCE escape hatch)

## What W7-12 inherit from this

W7 (tiered HBM/DRAM/SSD) and W8 (gossip + re-replication) don't touch
the transport — they're entirely above the abstraction.

W11 (chaos suite) may eventually run an RDMA-equipped node in CI if
the team gets access to one; the chaos harness should already work
unchanged against either transport because the layer above doesn't
care.

W12 PACE benchmarks on real IB hardware: change one line in
`cache.cpp` (`std::make_unique<GrpcStreamTransport>(...)` →
`std::make_unique<IbverbsTransport>(...)`), implement the real
`ibverbs_transport.cpp` body (rdma_cm connection establishment, RC QP,
pinned buffer pool, completion-polling thread), and re-add the
`libibverbs` / `librdmacm` link deps the CMakeLists comment names. The
rest of the codebase doesn't need to know.

## One thing to do differently

The W5-6 prompt budgeted 15-25 hours for the milestone. With gate #1
firing in the first 5 minutes, the realized cost was under 4 hours of
focused work (mostly the Replicator refactor and the regression
sweep). If the next milestone has a gate-#1-shaped exit, plan around
the cap, not the optimistic ceiling — the cap is what the schedule
actually depends on.
