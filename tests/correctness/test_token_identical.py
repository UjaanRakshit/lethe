"""Correctness: vLLM + Lethe must produce token-identical output to vanilla vLLM.

This is the W1 acceptance test. If this regresses at any week, fix it before
anything else. A cache that changes outputs is worse than no cache.
"""
from __future__ import annotations

import pytest

# from vllm import LLM, SamplingParams
# from lethe_client import LetheCacheConnector


PROMPTS = [
    "The capital of France is",
    "Once upon a time in a distant land,",
    "def fibonacci(n):\n    if n <= 1:\n",
]


@pytest.mark.skip("requires running Lethe cluster + GPU vLLM")
def test_outputs_match_vanilla_vllm():
    # vanilla = LLM(model=MODEL)
    # cached = LLM(model=MODEL, kv_transfer_config={
    #     "connector": LetheCacheConnector("localhost:50051"),
    # })
    # sp = SamplingParams(temperature=0, max_tokens=64)
    # for p in PROMPTS:
    #     a = vanilla.generate([p], sp)[0].outputs[0].token_ids
    #     b = cached.generate([p], sp)[0].outputs[0].token_ids
    #     assert a == b, f"diverged on prompt: {p!r}"
    pass
