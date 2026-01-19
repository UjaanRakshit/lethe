# W4 cleared as of 2026-05-26 22:13 EDT

# W4 readiness gate — pre-session environment + regression sweep

**Date:** 2026-05-26 (gate ran after W3 close)
**Status:** **W4 CLEARED** to start in a focused session

## Environment

| Check | Result |
|---|---|
| WSL2 responsive | ✓ `Linux UjaanLaptop 5.15.167.4-microsoft-standard-WSL2` |
| Clock / TZ | ✓ `Tue May 26 22:06:57 EDT 2026` (America/New_York, -0400). Matches Windows side, no drift. |
| Disk on build root | ✓ `874G free / 1007G total` (9% used) — ample for 3-node integration temp files |
| `.venv-vllm` activates | ✓ Python 3.10.12 at `/home/syclone/lethe/.venv-vllm/bin/python` |
| GPU visible | ✓ `True NVIDIA GeForce RTX 4060 Laptop GPU` |
| HF auth cached | ✓ `hf auth whoami` → `Logged in user: RakshitUjaan`. (Note: `huggingface-cli` is deprecated as of `hf` CLI 1.x; use `hf auth whoami` going forward.) |

Overall env: **GREEN.**

## Build + test sweep (timed)

| Stage | Result | Wall clock |
|---|---|---|
| `rm -rf build build-tsan` | clean | n/a |
| `scripts/build.sh` (clean cold) | ✓ exit 0, no -Werror warnings | ~2-3 min (not precisely captured; bash quoting ate the timer) |
| `ctest --test-dir build` | ✓ **4/4** passed | 0.04 s |
| `pytest tests/ client/tests/` (excl. token-identical) | ✓ **31 passed / 3 skipped** | 9.07 s |
| `cmake -B build-tsan -DLETHE_ENABLE_TSAN=ON -DCMAKE_CXX_COMPILER=clang++` | ✓ | 12.4 s |
| `cmake --build build-tsan -j` | ✓ | **15.6 s real** (4m 22s user; parallel) |
| `ctest --test-dir build-tsan` | ✓ **4/4** passed, zero TSan findings | 0.13 s |

3 pytest skips: 2 deferred-W1.4 placeholders (`test_failover.py`) + 1 explicit smoke-deferred-to-W1.4 (`test_llm_generate_smoke_deferred_to_W1_4`). All expected.

## W1.4 token-identical GPU regression

```
tests/correctness/test_token_identical.py::test_token_identical_three_way_control PASSED [100%]
============= 1 passed in 153.49s (0:02:33) =============
```

**PASS** in 2:33. W3 introduced no token-identical drift — the W1 baseline survives the routing scaffolding + Router-into-Lookup wiring. Verified against the same Gemma-3-1B / bfloat16 / 10-prompt × 3-run setup as the W1.4 closeout.

## File inventory for W4

### New files expected at W4 start (should NOT exist yet)

| File | Status |
|---|---|
| `tests/unit/test_replication.cpp` | ✓ missing (expected) |
| `tests/integration/test_three_node.cpp` | ✓ missing (expected) |
| `client/tests/test_three_node_python.py` | ✓ missing (expected) |
| `scripts/run_3node.sh` | ✓ missing (expected) |
| `docs/weekly/W3_W4.md` | ✓ missing (expected) |
| `docs/decisions/W3_checkpoint.md` | was missing; written this session as housekeeping (the W4 prompt referenced it). Brief 60-line doc summarizing what W3 shipped + verification at W3 close. |

### Files W4 will modify (should exist)

| File | Size | Last touched |
|---|---|---|
| `cache_server/src/replication.cpp` | 880 bytes (stub) | W1.2.6 era |
| `cache_server/src/main.cpp` | 10572 bytes | W1.2 |
| `cache_server/src/cache.cpp` | 8692 bytes | W3 (Router wiring) |
| `client/lethe_client/client.py` | 5621 bytes | W1.2 |
| `client/lethe_client/epoch_watcher.py` | 6359 bytes | W0 |

All present at sensible sizes.

### Headers + proto integrity

- `proto/lethe.proto` last touched at `00d4127` (W1.2) — proto is locked, no surprise edits since W0 lock-in.
- `cache_server/include/lethe/replication.hpp` — no commits since W1.2.6.
- `cache_server/include/lethe/kv_transport.hpp` — no commits since W1.2.6.
- `cache_server/include/lethe/cache.hpp` — last touched at W3 `9690e96` (Router member wiring; expected).
- `cache_server/include/lethe/membership.hpp` — last touched at W3 `8601013` (drop [[maybe_unused]]; expected).

No surprises. All header / proto state is consistent with where W3 left it.

## W4 verification cost estimates

Based on timings from this gate. Per-iteration verification matrix for W4:

| Stage | This gate's wall clock | W4 expected delta |
|---|---|---|
| Clean build | ~2-3 min | +30s for the new replication.cpp + 3 new test executables compile; estimate 3-4 min for the W4 build cycle |
| ctest (build/) | 0.04 s | +1-3 s when test_replication runs in-process gRPC fake (cheap), +5-30 s when test_three_node spins 3 subprocess servers per test (the dominant cost) → estimate 10-60 s |
| pytest sweep (no token-identical) | 9.07 s | +5-15 s for test_three_node_python.py (server spawns + RPCs) → estimate 15-25 s |
| TSan rebuild | 15.6 s real | +5-10 s for the new sources → estimate 20-30 s real |
| TSan ctest | 0.13 s | same per-test cost as build/ but TSan slowdown is 5-15× on threaded code; 3-node tests may take 30s-2min under TSan → estimate 30 s – 3 min |
| W1.4 token-identical regression | 153 s | ±10% — no W4 change to the W1 path is expected, but the regression has to run to confirm |

**Per-iteration full-verification matrix estimate: 7-13 minutes wall clock** (clean build + all four test suites + W1.4 regression). For ~3-5 debug iterations during W4 implementation, budget **30-65 minutes of verification time alone** on top of implementation work.

The wider W4 implementation scope (per the W3-W4 prompt) is ~6-10 hours of focused coding. Including verification: realistic W4 session is **8-13 hours of focused work** — consistent with the "12-18 hours for W3-W4 combined" original estimate, minus the ~3-4 hours W3 took.

## Readiness verdict

**W4 cleared to start.** Nothing blocking. Environment is green. Build sweep clean. W1.4 regression green. File inventory is exactly the W3 close state. Per-iteration verification is reasonable (~10 min) so W4's debug cycles won't be wall-clock bound.

## Housekeeping done this session

- Created `docs/decisions/W3_checkpoint.md` (the W4 prompt referenced it; it didn't exist; brief 60-line backfill).
- This file (`W4_readiness.md`).
- No code changes; no edits to anything except the two new docs.
- Build artifacts under `~/lethe/build/` and `~/lethe/build-tsan/` are warm — W4's first build will be incremental (a few seconds), not the cold ~2-3 min reported above.

## Tomorrow's W4 session: starting state

- HEAD: `5860190 tests: include <unistd.h> in test_routing.cpp (POSIX ::write/::close)`
- Working tree: clean (after the two readiness docs commit).
- Build artifacts warm; venv ready; HF auth cached; GPU visible.
- W3-W4 prompt's "What W4 needs" list applies verbatim — see `W3_checkpoint.md` for the full breakdown.
