# W3 checkpoint — routing complete, W4 deferred to a follow-up session

**Date:** 2026-05-26
**Author:** W3-W4 session, checkpoint
**Status:** approved

## TL;DR

W3-W4 was bundled into one prompt with a 12–18 hour estimate. After
shipping W3 verified-clean, the implementer paused and surfaced the
session-capacity tradeoff rather than push shallowly through W4.
W3 closes here. W4 is its own session.

## What W3 shipped

| | What | Where |
|---|---|---|
| Routing | Consistent-hash router with BLAKE3 ring keys + first-8-bytes LE uint64 block-routing | `cache_server/src/routing.cpp` |
| BLAKE3 | Vendored 1.5.4 portable C (no SIMD dispatch), static lib `lethe_blake3` | `cache_server/third_party/blake3/` |
| Driver | `hash_compat_driver` CLI for cross-language tests | `tools/hash_compat_driver.cpp` |
| Wiring | Router constructed in LetheCache ctor; Lookup branches LocalHit / RemoteHit / Miss per primary ownership | `cache_server/src/cache.cpp` |
| Tests | C++ Router unit tests (5) + 22-vector cross-language hash compat | `tests/unit/test_routing.cpp` + `tests/correctness/test_hash_compat.py` |
| Types | `StaticPeer { node_id, address }` POD; Membership ctor widened | `cache_server/include/lethe/types.hpp` + `membership.hpp` |

## Verification at W3 close

- Clean build: pass, zero `-Werror` warnings (GCC + Clang via TSan path).
- ctest (build/): 4/4 pass.
- pytest (excl. W1.4 token-identical): 31 passed / 3 skipped.
- Cross-language hash compat (22 chained + 4 ring-key vectors): 5/5 pass.
- TSan build + ctest: 4/4 pass, zero TSan findings.
- W1.4 token-identical three-way control: PASS in 2:33 (W3 introduced no regression).

## What W4 needs (deferred)

- Replication: `replication.cpp` full impl (PeerClient gRPC wrapper, thread pool, async ReplicateOut, FetchFromAny, connection pool).
- Wiring: `main.cpp` peer parsing; `cache.cpp` async-Insert + read-repair Lookup.
- Python: `client.py` per-block primary routing + transparent fetch; `epoch_watcher.py` PeerStatus consumption.
- Tests: `tests/unit/test_replication.cpp` (in-process gRPC fake), `tests/integration/test_three_node.cpp`, `client/tests/test_three_node_python.py`.
- Infra: `scripts/run_3node.sh`, `deploy/docker-compose.yml` verify.
- Docs: `DECISIONS.md` entries for repl policy / connection pool / retry; `docs/weekly/W3_W4.md`.

## Why W4 was split

- W4 needs ~8-12 focused hours of new code + several long-running verifications (3-node integration cycles, TSan, W1.4 regression each iteration).
- Pushing shallowly through it would risk a stop-and-report condition firing late (e.g., async replication policy turning out untenable, read-repair semantics ambiguity) with the doc + verification skipped.
- The W3-W4 prompt explicitly allows checkpoint-and-resume ("you can save state and resume").
- Splitting at a verified-complete W3 → focused W4 session keeps W3 from being held hostage to W4 verification.

## Commits at W3 close (in order)

```
afd7698  W3: routing.cpp impl + vendored BLAKE3 + hash_compat_driver
c5d2d1d  build: enable C language for vendored BLAKE3 reference impl
8601013  build: drop [[maybe_unused]] on Membership members; wire BLAKE3 into tests
9690e96  W3: wire Router into LetheCache (ctor builds + Lookup consults it)
9bb126d  W3: test_routing.cpp (C++ ring tests + cross-lang driver smoke)
5860190  tests: include <unistd.h> in test_routing.cpp (POSIX ::write/::close)
```

## Note on this doc

This doc was written at the W3-readiness pass for W4 (the day after
the W3 checkpoint chat report). The original chat report served as
the live checkpoint; this is the persistent version so future
readers — or the W4 implementer — can ground in W3 state without
reading the conversation transcript.
