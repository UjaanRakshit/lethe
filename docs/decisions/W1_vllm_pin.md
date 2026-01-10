# W1 decision: pin `vllm==0.19.1`

**Date:** 2026-05-26
**Author:** session-zero gate (W1)
**Status:** approved
**Supersedes:** the previous `vllm>=0.6.0` constraint in
`client/pyproject.toml`.

## TL;DR

Pin exactly `vllm==0.19.1`. The V1 KV connector API
(`KVConnectorBase_V1`) is the integration surface; it is nominally
marked experimental but is in production use by LMCache, NIXL, and
Mooncake. The pin choice has API-churn risk that we explicitly accept
because Lethe is a 12-week project. For a multi-year deployment, this
risk would not be acceptable.

## Source-of-truth read

All line citations in this document refer to the source tarball
`vllm-0.19.1.tar.gz` from
[PyPI](https://pypi.org/pypi/vllm/0.19.1/json), uploaded 2026-04-18,
extracted to `c:\Users\ujaan\scratch_vllm\extract_0.19.1\vllm-0.19.1\`
and inspected on 2026-05-26. Wheels for this version ship only as
`cp38-abi3-manylinux_2_31_{aarch64,x86_64}.whl` — Python-only
connector code is identical between sdist and wheel.

The corresponding 0.18.1 and 0.19.0 tarballs were inspected
side-by-side. **0.19.1's `base.py` is byte-identical to 0.19.0's** (`diff`
empty); 0.19.0 differs from 0.18.1 only in the
`handle_preemptions` signature (see "Breaking changes" below).

## Candidate set

The original prompt named the 0.6.x / 0.7.x lines; this was based on
prior knowledge from when those were the current stable lines. As of
2026-05-26, PyPI lists vLLM up to 0.21.0. Pinning to 0.6/0.7 today
would mean targeting a connector API two major-iteration generations
old. The candidate set was revised — confirmed with the project owner
in chat — to focus on recent stable lines that still meet the
≥30-day age criterion:

| version | uploaded | age (days as of 2026-05-26) |
|---|---|---|
| 0.18.1 | 2026-03-31 | 56 |
| 0.19.0 | 2026-04-03 | 53 |
| 0.19.1 | 2026-04-18 | 38 |

0.20.0 (29d) and onward were excluded for failing the ≥30-day rule.
0.21.0 (11d) is too fresh — bug reports have not had time to settle.
0.17.x was not inspected because 0.18.x already covers "one minor
version older than the recommended pin."

## Four criteria, evaluated per candidate

The criteria, in priority order, as set in the W1 gate prompt:

1. Connector API stable, not flagged experimental.
2. At least one in-tree reference implementation we can mirror.
3. Released ≥30 days ago.
4. Compatible with CUDA 12.x, Hopper-class GPU.

| | 0.18.1 | 0.19.0 | 0.19.1 |
|---|---|---|---|
| 1. Stable / not experimental | ⚠ same warning as 0.19.x at `base.py:189-192` | ⚠ warning present | ⚠ warning present |
| 2. In-tree reference impl | ✓ 8 reference connectors (ExampleConnector, LMCache, mooncake/, NIXL, …) | ✓ same + `simple_cpu_offload_connector` | ✓ same as 0.19.0 |
| 3. ≥30 days old | ✓ 56 | ✓ 53 | ✓ 38 |
| 4. CUDA 12.x, Hopper | ✓ pins `torch==2.10.0` | ✓ same | ✓ same |

Criterion 1 ("not experimental") fails *softly* across all three
candidates. See "Experimental-warning trade-off" below for the honest
treatment of this.

## Breaking changes across the candidate set

`diff` of `base.py` 0.18.1 → 0.19.0 → 0.19.1 found exactly one delta
across all three versions:

```
0.18.1:  def handle_preemptions(self, preempted_req_ids: set[str]):
0.19.x:  def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
```

(`base.py:291` in both versions.) `handle_preemptions` is non-abstract
(has a default no-op body); only connectors that override it for async
saves (e.g., `OffloadingConnector`) are affected. Lethe does not plan
to override `handle_preemptions` in W1.

`base.py` is byte-identical between 0.19.0 and 0.19.1.

## Pin choice and why

**Pinned to `vllm==0.19.1`.**

- vs. 0.19.0: 0.19.1 is the bug-fix release for the same byte-identical
  abstract surface. Always pick the bug-fix.
- vs. 0.18.1: 0.19.x's `handle_preemptions` widening is a deliberate
  API improvement aligning with where the project is going; targeting
  0.18.1 means we'd take that change at some future upgrade anyway.
  0.19.x has the broader reference-connector set as well
  (`simple_cpu_offload_connector` added).
- All four criteria satisfied (criterion 1 partial; see below).
- 38d age is comfortably past the floor; recent enough that the
  reference impls remain fresh, not so recent that bug reports are
  unsettled.

## Experimental-warning trade-off

`base.py:189-192` of every candidate emits, on every connector
instantiation:

> `"Initializing KVConnectorBase_V1. This API is experimental and
> subject to change in the future as we iterate the design."`

This is a literal fail on criterion 1 as written. The honest treatment:

- The V1 connector API has carried this warning for roughly 18 months
  across many vLLM minor releases.
- LMCache, NIXL, and Mooncake — all production deployments — build
  against this exact API. Their published configurations target V1.
- The "experimental" marker reflects the vLLM project's intent to
  reserve the right to evolve the API, not a stability indictment of
  the current shape.
- Refusing to pin against an API marked experimental would mean
  refusing to integrate with vLLM at all. The next-most-stable
  connector API is the legacy `KVConnectorBase` (v0), which is
  reachable from `vllm.distributed.kv_transfer.kv_connector.base` —
  it's a 370-byte stub at this point and lacks all the
  scheduler-side primitives we need.

**What we accept by pinning to an experimentally-flagged API:**

- A future vLLM release (e.g. 0.22.0) could ship `KVConnectorBase_V2`
  with breaking changes. If it does, the rework cost for Lethe is a
  rewrite of `client/lethe_client/vllm_hook.py`. That file is
  ~150 lines after W1.1; the cost is bounded.
- For a 12-week project ending at W12, the probability that a V2
  shipping breaks us *during* the project is low, and the recovery
  cost if it does is one focused W1.x-shaped session.
- For a multi-year production deployment, this trade-off would not
  be acceptable. The right answer in that context would be to
  vendor a frozen copy of the connector module, or pin to whatever
  version Mooncake/LMCache are themselves pinning at the time of
  deployment.

We accept the risk explicitly, with eyes open.

## What would force a re-pin

The conditions under which we'd revisit this decision:

1. **vLLM ships V2 connector API.** `KVConnectorBase_V2` lands; our
   `KVConnectorBase_V1` subclass stops working. Rewrite `vllm_hook.py`
   to target V2; bump pin to whichever vLLM version first ships V2 as
   the default.
2. **Lethe goes multi-tenant.** Out of W0 scope per CLAUDE.md
   non-goals, but if added later, the connector likely needs
   per-tenant isolation that the V1 API doesn't expose. Re-evaluate.
3. **A security fix lands in a later 0.19.x.** Bump to that patch,
   verify `base.py` still byte-identical to 0.19.1, no other
   action.
4. **We need a feature only present in 0.20.x+.** E.g., HMA support
   (the `SupportsHMA` mixin) becomes mandatory rather than optional.
   Re-pin to a 0.20.x or later release that has had 30 days of
   bug reports to settle.

Re-pin events should append a new entry to `docs/DECISIONS.md`
referencing this file and stating which trigger above applied.

## Where to read the contract

For the W1 implementer (and anyone touching `vllm_hook.py` going
forward):

| What | Where in the wheel |
|---|---|
| Abstract base class | `vllm/distributed/kv_transfer/kv_connector/v1/base.py:170` (`class KVConnectorBase_V1`) |
| Constructor signature to match exactly | `base.py:183-207` |
| Required abstract methods (worker side) | `base.py:298-361` (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`) |
| Required abstract methods (scheduler side) | `base.py:449-518` (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_meta`) |
| `KVConnectorRole` enum | `base.py:123-128` |
| `KVConnectorMetadata` base | `base.py:140-146` (subclass it for `LetheConnectorMetadata`) |
| Connector registry / loader | `vllm/distributed/kv_transfer/kv_connector/factory.py:27-131` |
| Out-of-tree registration mechanism | `factory.py:112-123` — set `kv_transfer_config.kv_connector_module_path` to `"lethe_client.vllm_hook"` and `kv_connector` to `"LetheCacheConnector"` |
| Canonical reference impl | `v1/example_connector.py` (450 LOC) |
| Mooncake-shaped impl (architecturally closest to Lethe) | `v1/mooncake/mooncake_connector.py` |

## Acceptance criterion for the pin itself (not the connector impl)

Adding here so it's documented in a single place:

After the pin lands in `client/pyproject.toml`:

```python
# A check that should pass on any machine with vllm==0.19.1 installed,
# even before vllm_hook.py is rewritten. Confirms the pin is honored
# and the V1 base class is importable.
import vllm
assert vllm.__version__ == "0.19.1", vllm.__version__
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1, KVConnectorRole,
)
```

The connector loadability check (factory round-trip) is the W1.1
acceptance criterion, documented in `docs/weekly/W0.md` and the
session-end report for W1.1.
