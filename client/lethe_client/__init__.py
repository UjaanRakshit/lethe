"""Lethe Python client and vLLM integration.

`LetheCacheConnector` is imported lazily via PEP 562 ``__getattr__`` so
that other consumers of this package (the chaos suite, benchmark
harnesses, the C++ server's own integration tests) can use
``LetheClient`` / ``chained_block_hash`` without installing vLLM. The
connector hard-imports ``vllm`` at module load time; importing it
eagerly would force every consumer to depend on vLLM transitively.
"""

from typing import TYPE_CHECKING

from .client import LetheClient
from .routing import chained_block_hash, prefix_block_ids

if TYPE_CHECKING:
    # Make static type checkers see the symbol; runtime resolution
    # still goes through __getattr__ below.
    from .vllm_hook import LetheCacheConnector

__all__ = [
    "LetheClient",
    "LetheCacheConnector",
    "chained_block_hash",
    "prefix_block_ids",
]
__version__ = "0.1.0"


def __getattr__(name: str):
    # PEP 562 module-level __getattr__: lazy-import vllm_hook only when
    # someone actually asks for LetheCacheConnector. Failing fast at this
    # boundary (rather than at package import) means missing-vllm errors
    # land at the call site that wanted vLLM, not at every importer.
    if name == "LetheCacheConnector":
        from .vllm_hook import LetheCacheConnector

        return LetheCacheConnector
    raise AttributeError(f"module 'lethe_client' has no attribute {name!r}")
