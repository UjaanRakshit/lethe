"""Disaggregated prefill/decode — single-engine role sequencing.

HONEST SCOPE NOTE. This is NOT physical two-instance disaggregation
(separate prefill and decode GPU workers). It is ONE vLLM engine driven
in two role-sequenced phases against a Lethe cluster:

  Phase 1 (prefill role): run prompt P through the engine. The
    LetheCacheConnector's save path exports P's KV blocks to Lethe.
  Phase 2 (decode role): run the SAME prompt P again. The connector's
    get_num_new_matched_tokens finds P's blocks in Lethe, start_load_kv
    pulls them into the paged-KV buffer, and decode proceeds WITHOUT
    recomputing the prefix.

The KV genuinely round-trips through Lethe between the phases — that is
the disaggregated KV TRANSPORT PATH being validated. Physical worker
separation (two engines, two GPUs) is deferred until hardware with
48-80 GB VRAM makes a real prefill/decode split trivial. Until then,
naming things "prefill_worker process" / "decode_worker process" would
be dishonest — there is one process, one engine, two phases.

CRITICAL CONFIG. The engine MUST be built with
`enable_prefix_caching=False`. With native prefix caching ON, vLLM would
cache P's KV after phase 1 and serve phase 2 from its own cache, so the
decode phase would never touch Lethe — a false green. Disabling it makes
Lethe the only external-cache path. (Verified in vllm 0.19.1: the
scheduler calls the connector's get_num_new_matched_tokens whenever a
connector is configured and the request has zero locally-computed
tokens; with caching off, local is always zero, so Lethe is consulted.)

This module is import-light: it does NOT import vllm at module scope, so
it can be imported by tooling that only wants the orchestration shape.
The engine is passed in already-built (the test's child process builds
it with the connector configured).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseResult:
    """Outcome of one role phase."""

    token_ids: list[int]
    prompt_token_ids: list[int]
    wall_seconds: float


@dataclass
class DisaggResult:
    """Full prefill→decode round-trip outcome for one prompt."""

    prompt: str
    prefill: PhaseResult
    decode: PhaseResult
    # Lethe block hashes for the prompt prefix (whole blocks), derived
    # from the EXACT token_ids vLLM used so they match the connector's
    # stored hashes bit-for-bit.
    prefix_block_hashes: list[bytes] = field(default_factory=list)


class RoleSequencedDisagg:
    """Drives one vLLM engine through prefill then decode phases.

    The engine must already have the LetheCacheConnector configured and
    `enable_prefix_caching=False`. See module docstring.
    """

    def __init__(self, llm: Any, block_size: int = 16) -> None:
        self._llm = llm
        self._block_size = block_size

    def prefill(self, prompt: str) -> PhaseResult:
        """Phase 1: run P through the engine so the connector exports
        P's KV to Lethe.

        max_tokens: vLLM 0.19.1 requires max_tokens >= 1 (a 0-token
        SamplingParams is rejected at validation). We use 1 and discard
        the single decoded token — the prefill computes (and the
        connector saves) the WHOLE prompt's KV regardless of how many
        tokens we then decode. The saved KV is the prompt prefix, which
        is all the decode phase needs.
        """
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
        t0 = time.monotonic()
        out = self._llm.generate([prompt], sp)
        wall = time.monotonic() - t0
        o = out[0]
        return PhaseResult(
            token_ids=list(o.outputs[0].token_ids),
            prompt_token_ids=list(o.prompt_token_ids or []),
            wall_seconds=wall,
        )

    def decode(self, prompt: str, max_tokens: int = 64) -> PhaseResult:
        """Phase 2: run the SAME P again. The connector loads P's KV
        from Lethe (no prefix recompute) and decode generates."""
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
        t0 = time.monotonic()
        out = self._llm.generate([prompt], sp)
        wall = time.monotonic() - t0
        o = out[0]
        return PhaseResult(
            token_ids=list(o.outputs[0].token_ids),
            prompt_token_ids=list(o.prompt_token_ids or []),
            wall_seconds=wall,
        )

    def prefix_block_hashes(self, prompt_token_ids: list[int]) -> list[bytes]:
        """Chained BLAKE3 block hashes for the whole-block prefix, using
        the EXACT token_ids vLLM used (so they match the connector's
        stored BlockId hashes). Mirrors the connector's hash chain.
        """
        from lethe_client.routing import chained_block_hash

        n_blocks = len(prompt_token_ids) // self._block_size
        out: list[bytes] = []
        running = b"\x00" * 32
        for i in range(n_blocks):
            start = i * self._block_size
            block_tokens = prompt_token_ids[start : start + self._block_size]
            running = chained_block_hash(running, block_tokens)
            out.append(running)
        return out

    def run(self, prompt: str, max_tokens: int = 64) -> DisaggResult:
        """Prefill then decode for one prompt; bundle the result."""
        prefill = self.prefill(prompt)
        decode = self.decode(prompt, max_tokens=max_tokens)
        hashes = self.prefix_block_hashes(prefill.prompt_token_ids)
        return DisaggResult(
            prompt=prompt,
            prefill=prefill,
            decode=decode,
            prefix_block_hashes=hashes,
        )
