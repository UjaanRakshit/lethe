"""Prefill-only vLLM worker (W9).

Spins up a vLLM engine in prefill-only mode (max_tokens=0) with the
LetheCacheConnector configured to export KV blocks after prefill.

Run:
    python -m disagg.prefill_worker --port 8001 --lethe localhost:50051 \
        --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import logging

# from vllm import LLM, SamplingParams
# from lethe_client import LetheCacheConnector

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--lethe", default="localhost:50051")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--block-size", type=int, default=16)
    args = p.parse_args()

    # TODO(W9):
    #   connector = LetheCacheConnector(args.lethe, block_size=args.block_size)
    #   llm = LLM(model=args.model, kv_transfer_config={"connector": connector})
    #   run_http_server(llm, port=args.port)
    raise NotImplementedError


if __name__ == "__main__":
    main()
