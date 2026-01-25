# W7 checkpoint — tiered storage complete, W8 deferred to follow-up

**Date:** 2026-05-27
**Status:** W7 closed clean. W8 (eviction loop + real heartbeats + failover) is its own focused session per the W7-W8 prompt's authorized internal checkpoint.

## What shipped

| | What | Where |
|---|---|---|
| SSD storage | mmap'd slot allocator, 64 KiB slots, magic-byte liveness sentinel, in-memory index rebuilt at startup | `cache_server/{include/lethe,src}/ssd_block_store.{hpp,cpp}` |
| Tier composition | HBM (optional, default disabled until LETHE_ENABLE_CUDA) + DRAM (always) + SSD (configurable). Tier-hint fallthrough on Put, promotion-on-Get past threshold, explicit Demote for the W8 evictor | `cache_server/src/tiered_store.cpp` |
| Lifetime contract | `LookupResult::Entry::local_data` switched from `span<byte>` (W1 borrowed) to `vector<byte>` (W7 owned). SSD can't lend spans into mmap'd slots across reuse | `cache_server/include/lethe/cache.hpp:69-87` |
| Wire-through | `cache.cpp::Insert` respects `tier_hint`; idempotence via used-bytes delta across all tiers (matches W4 contract that `test_repeated_insert_is_idempotent` enforces) | `cache_server/src/cache.cpp` |
| Production defaults | main.cpp sets 256 MiB SSD at `/tmp/lethe-<node_id>/ssd` so smoke runs exercise the SSD path | `cache_server/src/main.cpp` |
| Unit tests | 8 scenarios: two-tier basic, hint fallthrough, promotion threshold, DRAM→SSD demote, Erase wipes access counts, oversized rejection, snapshot, SSD persistence across destruct | `tests/unit/test_tiered_store.cpp` |

## CHECKPOINT 1 — acceptance evidence

| Check | Status |
|---|---|
| Build clean LETHE_ENABLE_RDMA=OFF (default) | ✓ zero -Werror |
| Build clean LETHE_ENABLE_RDMA=ON (link smoke) | ✓ zero -Werror |
| ctest in build/ | ✓ 5/5 (including new test_tiered_store) |
| ctest in build-tsan/ | ✓ 5/5, zero TSan findings |
| pytest hash_compat (W3 invariant) | ✓ 5/5 |
| pytest client_roundtrip (W1.2 incl. idempotence) | ✓ 5/5 |
| pytest 3-node multi-node (W4) | ✓ 4/4 |
| pytest W1.4 token-identical | ✓ wall=120.6s, all_identical=True, gen-time symmetry A/B/C 14.6/14.4/14.1 |

Acceptance A through D (W7 portion of the W7-W8 prompt):
- **A.** Build both flavors clean. ✓
- **B.** SSD persistence: process destroy → reconstruct → block readable. `TestSsdPersistenceAcrossDestruct` ✓
- **C.** Tier transitions: promotion at threshold ✓, demotion before drop ✓, access_counts_ wiped on Erase ✓
- **D.** W1+W3+W4 regressions stay green. ✓ all of them

## What W8 inherits

The Evictor surface (`eviction.hpp`) is unchanged from W4; the .cpp is still a stub. W8 work:
- SIEVE eviction loop (3 threads, one per tier).
- High/low watermark trigger + scan-interval default 500ms.
- EvictBroadcast gossip RPC.
- Heartbeat thread + failure detection in `membership.cpp` (real impl over the W3-W4 static-list stub).
- Re-replication on membership change in `replication.cpp` (the `TriggerReReplication` body).
- `test_eviction.cpp` and `tests/integration/test_failover_recovery.py`.

All five interfaces already exist in the headers; W8 is filling in the bodies. The TieredStore::Demote method is ready to be called by the SIEVE loop.

## Execution-call records (carried into W8)

- **SSD slot size: 64 KiB.** Configurable via `TieredStoreConfig::ssd_slot_bytes`. Sized to fit a typical KV block + header on a single page-aligned cache line worth of metadata. The fragmentation-resistant variant (variable-size or buddy allocator) is W11-chaos-suite scope if the simple allocator surfaces fragmentation under realistic workload.
- **No fsync.** Per the W0 design — host crash loses recent SSD writes; process restart survives via page cache (acceptance test confirms). When `msync` is wanted for hard durability, it goes on the same write-path with the magic-byte-last invariant the file already encodes.
- **Owned bytes uniformly.** `GetResult.data` is `vector<byte>`. The HBM/DRAM tiers technically *could* still lend spans, but uniform ownership beats per-tier dispatch — one memcpy per Get is negligible at 64 KiB.

## Commits this checkpoint

```
b1c99fd  W7: tiered storage with SSD persistence via mmap'd slot allocator
92fbe10  W7: fix SsdSlotHeader alignment — reorder so inserted_epoch is 8-aligned
388c0e5  W7: restore Insert idempotence + silence clang unused-private-field
```

## Verification cost (W7 portion)

| Stage | Wall |
|---|---|
| Required-context reading (headers, ARCHITECTURE.md, existing stubs) | ~25 min |
| SsdBlockStore (header + impl) | ~45 min |
| TieredStore tier-aware rewrite + GetResult/Entry owned-bytes lift | ~40 min |
| cache.cpp + main.cpp wire-up | ~15 min |
| test_tiered_store.cpp (8 scenarios incl. persistence) | ~25 min |
| Build iteration (alignment fix, idempotence fix, -Werror fix) | ~20 min |
| Regression sweep (ctest + TSan + pytest sweep + W1.4) | ~5 min |
| Docs (this file) | ~10 min |
| **Total** | **~3:05** |

Well inside the 6-10 hour W7 estimate. The fall-through fix-ups (alignment, idempotence, -Werror) each took 5-10 minutes — exactly the kind of small mid-flight corrections the regression sweep is designed to catch.

## Why pause here

The W7-W8 prompt explicitly authorizes the W7-close → W8-start checkpoint:
> "Single continuous session, full autonomy, natural checkpoint authorized between W7 close and W8 start."

W8 is independently estimated 12-18 hours and is concurrency-heavy (heartbeat threads + eviction threads + re-replication thread pool + the existing gRPC threadpool, all needing TSan cleanliness). Pushing through it on the same context that's already at ~3 hours of W7 is the same shallow-execution risk the W3-W4 split warned against. Better to fork a focused W8 session.

## Next milestone readiness

**W7 verified, ready for W8 follow-up.**

W8 starts from this commit (`388c0e5`). The required context for W8 is the same set the W7-W8 prompt named, plus this checkpoint as the "what's already there" baseline. The W8 acceptance items (E-I in the prompt) are the gate; CHECKPOINT 2 (failover recovery 3.5s budget) is the load-bearing one.
