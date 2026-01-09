"""Disaggregated prefill/decode orchestrator (W9).

DistServe-style architecture:
  - Prefill workers process incoming prompts and produce KV blocks.
  - Lethe stores the KV blocks (sharded by prefix-chained hash).
  - Decode workers fetch the KV blocks from Lethe and continue generation.

This orchestrator is the front door: it receives requests, schedules them
onto a prefill worker, waits for prefill completion + KV insert, then routes
the decode portion to a decode worker. The decode worker's vLLM connector
pulls the KV blocks from Lethe transparently.

W9 milestone: end-to-end token generation through two vLLM instances with
Lethe as the KV transport.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from lethe_client import LetheClient

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    prompt: str
    max_tokens: int = 256
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class WorkerEndpoint:
    name: str
    address: str            # host:port for the vLLM HTTP server
    role: str               # "prefill" or "decode"
    in_flight: int = 0
    max_concurrent: int = 8


class DisaggOrchestrator:
    def __init__(
        self,
        prefill_workers: list[WorkerEndpoint],
        decode_workers: list[WorkerEndpoint],
        lethe_address: str,
    ):
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.lethe = LetheClient(primary_address=lethe_address)

    async def generate(self, req: GenerationRequest) -> str:
        prefill = self._pick_least_loaded(self.prefill_workers)
        decode = self._pick_least_loaded(self.decode_workers)
        logger.info(
            "req=%s prefill=%s decode=%s", req.request_id, prefill.name, decode.name
        )

        # 1) Prefill: worker runs prompt through transformer, writes KV
        #    blocks into Lethe via the connector hook (W1).
        await self._invoke_prefill(prefill, req)

        # 2) Decode: worker connects to Lethe, fetches KV for the prompt
        #    tokens, then begins autoregressive decoding.
        return await self._invoke_decode(decode, req)

    async def _invoke_prefill(self, w: WorkerEndpoint, req: GenerationRequest):
        # TODO(W9): POST to w.address with prompt, max_tokens=0, kv_export=True.
        raise NotImplementedError

    async def _invoke_decode(self, w: WorkerEndpoint, req: GenerationRequest) -> str:
        # TODO(W9): POST to w.address with prompt (so it knows the prefix
        #    hash chain), max_tokens=req.max_tokens, kv_import=True.
        raise NotImplementedError

    @staticmethod
    def _pick_least_loaded(pool: list[WorkerEndpoint]) -> WorkerEndpoint:
        return min(pool, key=lambda w: w.in_flight / max(1, w.max_concurrent))


async def _demo():
    orch = DisaggOrchestrator(
        prefill_workers=[WorkerEndpoint("p0", "http://localhost:8001", "prefill")],
        decode_workers=[WorkerEndpoint("d0", "http://localhost:8101", "decode")],
        lethe_address="localhost:50051",
    )
    out = await orch.generate(GenerationRequest(prompt="Hello, "))
    print(out)


if __name__ == "__main__":
    asyncio.run(_demo())
