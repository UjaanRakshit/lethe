"""Decode-only vLLM worker (W9).

Spins up a vLLM engine configured to fetch the KV cache from Lethe for any
incoming request's prefix tokens, then proceed with decoding.

Run:
    python -m disagg.decode_worker --port 8101 --lethe localhost:50051 \
        --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--lethe", default="localhost:50051")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--block-size", type=int, default=16)
    args = p.parse_args()

    # TODO(W9): mirror prefill_worker but with kv_import=True semantics.
    raise NotImplementedError


if __name__ == "__main__":
    main()
