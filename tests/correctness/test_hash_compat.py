"""Cross-language hash equivalence: Python `lethe_client.routing` MUST
produce byte-identical digests to the C++ cache server's BLAKE3 routines
for every `(prev_hash, tokens)` input AND every `f"{peer}#{vn}"` ring key.

If this regresses, the Python client will route to a different node than
the C++ server expects, and the cluster will silently corrupt its prefix
locality. This is the "mirror invariant" from client/CLAUDE.md.

How this test works:
  1. Compute digests in Python via `chained_block_hash` and the ring-key
     hash that `HashRing` uses internally.
  2. Shell out to `build/tests/hash_compat_driver` (a tiny C++ binary
     built alongside the unit tests; see cache_server/tests/CMakeLists)
     that hashes the same inputs and writes hex digests on stdout.
  3. Assert byte equality.
  4. Pin the Python side to known BLAKE3 test vectors so that a missing
     C++ driver doesn't let a Python regression slip through unnoticed.

When the C++ driver is missing (pre-W1, or `cmake` not run), this test
asserts only the BLAKE3 reference vectors and emits a warning rather
than passing silently. CI must run with the driver present.
"""
from __future__ import annotations

import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from lethe_client.routing import HashRing, chained_block_hash, _HASH


REPO_ROOT = Path(__file__).resolve().parents[2]
# The driver binary is built under cache_server/CMakeLists.txt and
# lands at build/cache_server/hash_compat_driver. (Earlier scaffold
# guessed build/tests/hash_compat_driver; corrected to the actual
# build location.)
DRIVER_PATH = REPO_ROOT / "build" / "cache_server" / "hash_compat_driver"


# BLAKE3 published test vectors. If `blake3` ever changes algorithm or
# the C++ side picks a different library, these change and we see it.
# Source: https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json
BLAKE3_REFERENCE_VECTORS: list[tuple[bytes, str]] = [
    (b"", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (
        bytes(range(1)),
        "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
    ),
    (
        bytes(range(63)),
        "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b",
    ),
]


def _ring_key_digest(peer: str, vnode: int) -> bytes:
    """The exact bytes HashRing.set_peers hashes for one virtual node."""
    return _HASH(f"{peer}#{vnode}".encode()).digest()


def test_blake3_reference_vectors_match():
    """Python side is pinned to upstream BLAKE3 vectors — failures here
    mean the `blake3` package changed under us."""
    for data, expected_hex in BLAKE3_REFERENCE_VECTORS:
        got = _HASH(data).digest().hex()
        assert got == expected_hex, (
            f"BLAKE3 vector regressed for input of len {len(data)}: "
            f"expected {expected_hex}, got {got}"
        )


def test_chained_block_hash_is_stable():
    """The chained-block-hash recipe is deterministic and stable across
    Python runs. Pin one concrete output so a future refactor that
    changes the input encoding gets caught."""
    prev = bytes(32)
    tokens = [1, 2, 3, 4, 5]
    h1 = chained_block_hash(prev, tokens)
    h2 = chained_block_hash(prev, tokens)
    assert h1 == h2
    assert len(h1) == 32
    # Layout sanity: tokens are packed as little-endian uint32.
    expected = _HASH(prev)
    expected.update(b"".join(struct.pack("<I", t) for t in tokens))
    assert h1 == expected.digest()


def test_hashring_ring_key_is_stable():
    """Ring-key format must stay `f"{peer}#{vn}"` UTF-8 little-endian
    uint64 — if anything in HashRing changes, the test fails before a
    real C++ comparison even runs."""
    ring = HashRing(peers=["node0", "node1"], vnodes_per_peer=4)
    # The first virtual node of node0 should match the formula directly.
    expected_hash = int.from_bytes(_ring_key_digest("node0", 0)[:8], "little")
    # Pull internal ring to verify the entry exists with the right hash.
    ring_pairs = [(h, p) for h, p in ring._ring if p == "node0"]
    assert any(h == expected_hash for h, _ in ring_pairs), (
        "HashRing did not hash 'node0#0' the way the spec says"
    )


@pytest.mark.skipif(
    not DRIVER_PATH.exists(),
    reason=(
        f"C++ hash_compat_driver not built at {DRIVER_PATH}. "
        "Run `cmake --build build --target hash_compat_driver` before this "
        "test in CI. The Python-side reference-vector test above remains "
        "active and is the floor of correctness."
    ),
)
def test_cpp_python_chained_hash_agree():
    """Hand the C++ driver `(prev_hex tokens...)` lines on stdin; expect
    one hex digest per line back. Both sides must produce identical
    bytes — this is the load-bearing cross-language assertion."""
    # ≥20 vectors per the W3-W4 acceptance: empty / single / multi-
    # block sizes, block_size boundaries (1, 15, 16, 17, 32, 33),
    # and large/edge uint32 token values. Each pair drives one
    # chained_block_hash call (one block-worth of tokens).
    cases: list[tuple[bytes, list[int]]] = [
        # Empty + minimal.
        (bytes(32), []),
        (bytes(32), [0]),
        (bytes(32), [1]),
        # Around the W1 block_size=16 boundary.
        (bytes(32), list(range(1))),
        (bytes(32), list(range(15))),
        (bytes(32), list(range(16))),
        (bytes(32), list(range(17))),
        (bytes(32), list(range(32))),
        (bytes(32), list(range(33))),
        # Distinct prev_hash values.
        (b"\xff" * 32, [0xDEADBEEF, 0xCAFEBABE]),
        (b"\x42" * 32, list(range(64))),
        (bytes(range(32)), list(range(64))),
        (bytes(range(31, -1, -1)), list(range(8))),
        # uint32 boundary values.
        (bytes(32), [0, 1, 0xFFFF, 0x10000, 0xFFFFFFFE, 0xFFFFFFFF]),
        # Large blocks.
        (bytes(32), list(range(256))),
        (bytes(32), list(range(1024))),
        # Some adversarial-looking inputs.
        (bytes(32), [42] * 16),
        (bytes(32), [0] * 64),
        (bytes(32), [0xFFFFFFFF] * 16),
        # Realistic-shaped tokens (mimics Gemma tokenizer values).
        (bytes(32), [2, 1542, 8743, 12345, 999, 0, 4]),
        (bytes(32),
         [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]),
        # A chain extension: hash this with a non-zero prev_hash.
        (
            bytes.fromhex(
                "deadbeefcafebabe0123456789abcdef"
                "fedcba9876543210babecafebeefdead"
            ),
            list(range(48)),
        ),
    ]
    assert len(cases) >= 20, f"need ≥20 cross-language vectors, have {len(cases)}"
    stdin_lines = []
    for prev, tokens in cases:
        stdin_lines.append(prev.hex() + " " + " ".join(str(t) for t in tokens))
    stdin = "\n".join(stdin_lines) + "\n"

    proc = subprocess.run(
        [str(DRIVER_PATH)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    cpp_digests = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert len(cpp_digests) == len(cases), (
        f"driver returned {len(cpp_digests)} digests for {len(cases)} cases:"
        f" {proc.stdout!r}"
    )
    for (prev, tokens), cpp_hex in zip(cases, cpp_digests):
        py_hex = chained_block_hash(prev, tokens).hex()
        assert py_hex == cpp_hex, (
            f"divergence on (prev={prev.hex()[:16]}…, tokens={tokens[:4]}…): "
            f"py={py_hex} cpp={cpp_hex}"
        )


@pytest.mark.skipif(not DRIVER_PATH.exists(), reason="driver not built")
def test_cpp_python_ring_key_agree():
    """Same as above but for the ring-key hash so the Router and the
    Python HashRing place the same vnode at the same ring position."""
    cases = [("node0", 0), ("node0", 127), ("alpha", 1), ("z", 42)]
    stdin = "\n".join(f"{p} {v}" for p, v in cases) + "\n"
    proc = subprocess.run(
        [str(DRIVER_PATH), "--mode=ring_key"],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    cpp_digests = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for (peer, vn), cpp_hex in zip(cases, cpp_digests):
        py_hex = _ring_key_digest(peer, vn).hex()
        assert py_hex == cpp_hex, (
            f"ring key divergence on {peer}#{vn}: py={py_hex} cpp={cpp_hex}"
        )
