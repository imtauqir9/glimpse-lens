"""Perceptual-hash dedup — drop visually-identical frames before spending
compute on them (CLIP) and storage on their thumbnails.

dHash: resize to 9x8 grayscale, compare each pixel to its right neighbour ->
a 64-bit fingerprint that survives compression artifacts and tiny jitters.
A frame is dropped when its Hamming distance to the PREVIOUS KEPT frame is
within DEDUP_MAX_DISTANCE — static slides, talking heads and pauses collapse
to one representative frame each, while any real visual change passes.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ..config import DEDUP_ENABLED, DEDUP_MAX_DISTANCE
from .frames import Frame


# Gradient hashes are blind to flat color (any uniform frame hashes the same),
# so a mean-luminance delta larger than this also counts as "changed".
_LUMA_DELTA = 10


def dhash(jpeg: bytes) -> tuple[int, float]:
    """(64-bit gradient hash, mean luminance 0-255)."""
    img = Image.open(io.BytesIO(jpeg)).convert("L").resize((9, 8), Image.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).tobytes().hex(), 16), float(px.mean())


def _same(a: tuple[int, float], b: tuple[int, float]) -> bool:
    return ((a[0] ^ b[0]).bit_count() <= DEDUP_MAX_DISTANCE
            and abs(a[1] - b[1]) <= _LUMA_DELTA)


def dedup(frames: list[Frame]) -> list[Frame]:
    """Sequentially drop frames too similar to the last kept one."""
    if not DEDUP_ENABLED or len(frames) < 2:
        return frames
    kept: list[Frame] = [frames[0]]
    last = dhash(frames[0].jpeg)
    for f in frames[1:]:
        h = dhash(f.jpeg)
        if not _same(h, last):
            kept.append(f)
            last = h
    return kept
