"""vLLM ↔ Lethe integration.

The hook intercepts vLLM's PagedAttention block allocation: before vLLM
materializes KV blocks for a sequence, we ask Lethe whether the blocks
(addressed by their prefix-chained content hash) are already cached. On
hit, the cached blocks are streamed in instead of being recomputed.

Two integration points:

  1. KVConnector subclass (vLLM ≥0.6 supports `--kv-transfer-config`):
     a clean plugin point used by Mooncake-vLLM-integration, LMCache, etc.

  2. Direct monkey-patch of PagedAttention for older vLLM. Avoid this
     unless the connector path is unavailable.

W1: implement option 1 against a single-node Lethe.
W3: route per-block to the correct owner.
W9: hook fires from both prefill and decode workers (disaggregated path).

Correctness invariant: with the connector ENABLED, model outputs must be
token-for-token identical to vanilla vLLM on the same seed. The
tests/correctness/test_token_identical.py harness asserts this.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from .client import BlockId, LetheClient
from .routing import chained_block_hash

logger = logging.getLogger(__name__)


class LetheCacheConnector:
    """vLLM KV transfer connector backed by Lethe.

    Integrate via vLLM:
        from vllm import LLM
        from lethe_client import LetheCacheConnector

        connector = LetheCacheConnector(
            lethe_address="lethe-node0:50051",
            block_size=16,
        )
        llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
                  kv_transfer_config={"connector": connector})
    """

    def __init__(
        self,
        lethe_address: str,
        block_size: int = 16,
        model_id: int = 0,
        num_layers: Optional[int] = None,
        num_head_groups: int = 1,
    ):
        self.client = LetheClient(primary_address=lethe_address)
        self.block_size = block_size
        self.model_id = model_id
        self.num_layers = num_layers
        self.num_head_groups = num_head_groups

    # ----- vLLM-facing API -------------------------------------------------

    def get_num_external_hits(
        self,
        token_ids: list[int],
        prefix_hash: Optional[bytes] = None,
    ) -> int:
        """Return the number of blocks Lethe has cached for this sequence.

        Called by vLLM scheduler before scheduling prefill.
        """
        # TODO(W1): compute chained block hashes, call self.client.lookup.
        ids = self._build_block_ids(token_ids, prefix_hash)
        # res = self.client.lookup(ids, request_id="...")
        # return len(res.hits)
        return 0

    def recv_kv_caches(
        self,
        token_ids: list[int],
        prefix_hash: Optional[bytes] = None,
    ) -> dict[int, bytes]:
        """Fetch cached KV blocks for the prefix tokens.

        Returns dict[block_index → packed KV bytes]. Vllm copies these into
        its block pool instead of recomputing them in the prefill kernel.
        """
        # TODO(W1): lookup + fetch from the appropriate owner.
        raise NotImplementedError

    def send_kv_caches(
        self,
        token_ids: list[int],
        kv_blocks: dict[int, bytes],
        prefix_hash: Optional[bytes] = None,
    ) -> None:
        """Insert newly-computed KV blocks into Lethe after prefill.

        Called by vLLM after prefill completes and the KV cache is populated.
        """
        # TODO(W1): build BlockIds, call self.client.insert.
        pass

    # ----- internals -------------------------------------------------------

    def _build_block_ids(
        self,
        token_ids: list[int],
        prefix_hash: Optional[bytes] = None,
    ) -> list[BlockId]:
        """Generate the prefix-chained block IDs for a sequence."""
        ids: list[BlockId] = []
        running = prefix_hash or b"\x00" * 32
        for start in range(0, len(token_ids), self.block_size):
            block_tokens = token_ids[start : start + self.block_size]
            running = chained_block_hash(running, block_tokens)
            for layer in range(self.num_layers or 1):
                for hg in range(self.num_head_groups):
                    ids.append(
                        BlockId(
                            hash=running,
                            layer=layer,
                            head_group=hg,
                            model_id=self.model_id,
                        )
                    )
        return ids
