"""Bumper OCR using EasyOCR.

Reads team numbers off robot bumpers in match video. Constrained to digits
since FRC team numbers are 1-5 digit integers; this is faster and much more
accurate than general-purpose OCR.

Cropping: we hand EasyOCR the FULL robot bbox (not just the bumper region).
The library's text-detection sub-model finds the bumper area on its own,
and empirically gives marginally higher hit rates than a hard crop.

GPU-accelerated via PyTorch; ~20ms per call on an RTX 3070 Ti after warmup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class BumperReader:
    """Cached EasyOCR reader for digit-only text recognition.

    Lazy-loads the EasyOCR model on first use (~12 sec initialization on
    GPU including model download on first install).

    Public methods:
        read_digits(image)  -> list of (text, conf) for all detected digit
                               strings in the image
    """

    def __init__(self, gpu: bool = True, lang: str = "en"):
        self._reader = None
        self._gpu = gpu
        self._lang = lang

    def _ensure_loaded(self) -> None:
        if self._reader is not None:
            return
        # Late import — EasyOCR pulls in scikit-image and a few other heavy
        # deps. Importing it only at use time keeps `from detection.ocr import
        # ...` cheap for callers who don't need OCR.
        import easyocr
        self._reader = easyocr.Reader([self._lang], gpu=self._gpu, verbose=False)

    def read_digits(self, image: np.ndarray) -> list[tuple[str, float]]:
        """Return a list of (digit_string, confidence) for every text region
        EasyOCR finds in the image. Restricted to digits via `allowlist`."""
        if image is None or image.size == 0:
            return []
        self._ensure_loaded()
        result = self._reader.readtext(
            image,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )
        # result entries are [bbox, text, conf]
        return [(str(text), float(conf)) for (_box, text, conf) in result]


def crop_robot(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
    """Crop a robot from a frame using its bbox (in source-pixel coords).

    Returns an empty array if the bbox is fully out-of-bounds.
    """
    H, W = frame.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(W, int(bbox[2]))
    y2 = min(H, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def team_number_from_team_key(team_key: str) -> str:
    """TBA team keys look like "frc4499" — strip the prefix to get the number."""
    return team_key.removeprefix("frc")
