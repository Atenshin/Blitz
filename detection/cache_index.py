"""Fast time -> FrameDetections lookup over a MatchDetectionCache.

The GUI fires position_changed many times per second; for ~1000 cached frames
we want O(log n) lookup. The cache is already sorted by sec, so a bisect
suffices.
"""
from __future__ import annotations

import bisect
from pathlib import Path

from .schema import FrameDetections, MatchDetectionCache


class CacheIndex:
    """Wraps a MatchDetectionCache with a sec -> FrameDetections lookup."""

    def __init__(self, cache: MatchDetectionCache):
        self.cache = cache
        # Pre-extract the sec column so bisect doesn't pay attribute access
        # cost on every lookup.
        self._secs: list[float] = [f.sec for f in cache.frames]

    @property
    def frames(self) -> list[FrameDetections]:
        return self.cache.frames

    def find_frame_at(self, sec: float, max_lookahead: float = 0.5) -> FrameDetections | None:
        """Return the FrameDetections whose timestamp is closest to `sec`.

        Returns None if no cached frame is within `max_lookahead` seconds of
        the requested time. This prevents stale detections from a long-ago
        frame leaking into a moment where nothing was actually cached
        (e.g. between scoreboard transitions that we sampled around).
        """
        if not self._secs:
            return None
        idx = bisect.bisect_left(self._secs, sec)
        # idx is the first index with sec >= target; check it and the one before.
        candidates: list[int] = []
        if idx < len(self._secs):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(self._secs[i] - sec))
        if abs(self._secs[best] - sec) > max_lookahead:
            return None
        return self.cache.frames[best]


def load_cache_for_video(
    video_path: Path, detections_root: Path
) -> CacheIndex | None:
    """Return the cache for a given video, or None if no cache exists."""
    from .inference import cache_path_for
    cache_path = cache_path_for(video_path, detections_root)
    if not cache_path.exists():
        return None
    cache = MatchDetectionCache.read(cache_path)
    return CacheIndex(cache)
