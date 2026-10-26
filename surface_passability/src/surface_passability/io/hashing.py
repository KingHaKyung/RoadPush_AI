"""Stable content hashing for production artifacts."""

from __future__ import annotations

import hashlib

import numpy as np


def hash_mask(mask: np.ndarray) -> str:
    """Return the frozen SHA-256 digest of a mask's uint8 byte content."""

    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def hash_mask_after_uint8_cast(mask: np.ndarray) -> str:
    """Hash mask bytes after the caller-visible ``astype`` conversion step."""

    return hashlib.sha256(
        np.ascontiguousarray(mask.astype(np.uint8)).tobytes()
    ).hexdigest()


def hash_binary_mask(mask: np.ndarray) -> str:
    """Hash a boolean mask together with its exact shape."""

    packed = np.packbits(np.asarray(mask, dtype=bool), axis=None)
    shape = np.asarray(mask.shape, dtype=np.int64).tobytes()
    return hashlib.sha256(shape + packed.tobytes()).hexdigest()
