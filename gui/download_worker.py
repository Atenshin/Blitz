"""QThread-friendly wrapper around the downloader functions.

The worker exposes three modes — single URL, playlist+event, event-only — and
emits Qt signals for progress and completion. The actual download/TBA logic
lives in api/tba.py and downloader/youtube.py and is unchanged from M1.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

import yt_dlp

from api.tba import Match, TBAClient
from downloader.matcher import parse_title
from downloader.youtube import fetch_playlist_entries


class DownloadWorker(QObject):
    """Runs in a QThread. Emit start_*() to kick off; listen on signals."""

    progress = pyqtSignal(str, float)         # (message, percent 0-100; -1 = indeterminate)
    log = pyqtSignal(str)                     # human-readable log line
    finished = pyqtSignal(int, int, list)     # (succeeded, failed, error_messages)

    def __init__(
        self,
        videos_root: Path,
        format_spec: str,
        retries: int,
        tba_auth_key: str,
    ):
        super().__init__()
        self.videos_root = videos_root
        self.format_spec = format_spec
        self.retries = retries
        self.tba = TBAClient(tba_auth_key, cache_dir=videos_root)

    # --- public entry points (called via signals/slots from the dialog) ---

    def run_single(self, url: str, match_key: str | None) -> None:
        try:
            if match_key:
                out = self._video_path(match_key)
                label = match_key
            else:
                vid = _video_id_from_url(url)
                out = self.videos_root / "_unassigned" / f"{vid}.mp4"
                label = f"_unassigned/{vid}"
            self._download(url, out, label, total_items=1, item_index=1)
            self.finished.emit(1, 0, [])
        except Exception as e:  # surface all yt-dlp / network errors
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    def run_playlist(self, playlist_url: str, event_key: str) -> None:
        try:
            self.log.emit(f"[tba] fetching matches for {event_key}…")
            matches = self.tba.event_matches(event_key)
            matches_by_key = {m.key: m for m in matches}
            self.log.emit(f"[tba] {len(matches)} matches loaded")

            self.log.emit(f"[yt]  enumerating playlist…")
            entries = fetch_playlist_entries(playlist_url)
            self.log.emit(f"[yt]  {len(entries)} videos found")

            targets: list[tuple[Match, str]] = []
            unmatched: list[str] = []
            for entry in entries:
                parsed = parse_title(entry.title)
                if parsed is None:
                    unmatched.append(entry.title)
                    continue
                key = parsed.key(event_key)
                m = matches_by_key.get(key)
                if m is None:
                    unmatched.append(entry.title)
                    continue
                targets.append((m, entry.url))

            self.log.emit(f"[map] {len(targets)} mapped, {len(unmatched)} unmatched")
            self._download_targets(targets)
        except Exception as e:
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    def run_event(self, event_key: str) -> None:
        try:
            self.log.emit(f"[tba] fetching matches for {event_key}…")
            matches = self.tba.event_matches(event_key)
            targets: list[tuple[Match, str]] = []
            for m in matches:
                yt = next((v for v in m.videos if v.type == "youtube" and v.key), None)
                if yt and yt.youtube_url:
                    targets.append((m, yt.youtube_url))
            self.log.emit(f"[tba] {len(matches)} matches, {len(targets)} have YouTube links")
            self._download_targets(targets)
        except Exception as e:
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    # --- internals ---

    def _video_path(self, match_key: str) -> Path:
        event_key = match_key.split("_", 1)[0]
        return self.videos_root / event_key / f"{match_key}.mp4"

    def _download_targets(self, targets: list[tuple[Match, str]]) -> None:
        succeeded = 0
        errors: list[str] = []
        for i, (match, url) in enumerate(targets, 1):
            out = self._video_path(match.key)
            if out.exists():
                self.log.emit(f"[{i}/{len(targets)}] skip  {match.key} (exists)")
                succeeded += 1
                continue
            try:
                self._download(url, out, match.key, total_items=len(targets), item_index=i)
                succeeded += 1
            except Exception as e:
                errors.append(f"{match.key}: {e}")
                self.log.emit(f"[{i}/{len(targets)}] FAIL  {match.key}: {e}")
        self.finished.emit(succeeded, len(errors), errors)

    def _download(self, url: str, out: Path, label: str, total_items: int, item_index: int) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return
        out_no_ext = out.with_suffix("")

        def hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                downloaded = d.get("downloaded_bytes") or 0
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                pct = (downloaded / total * 100.0) if total else -1.0
                # Scale to overall batch progress so the bar moves smoothly.
                overall = ((item_index - 1) + (max(pct, 0) / 100.0)) / max(total_items, 1) * 100.0
                self.progress.emit(f"[{item_index}/{total_items}] {label}", overall)
            elif status == "finished":
                self.log.emit(f"[{item_index}/{total_items}] merging {label}…")

        ydl_opts = {
            "outtmpl": str(out_no_ext) + ".%(ext)s",
            "format": self.format_spec,
            "merge_output_format": "mp4",
            "retries": self.retries,
            "concurrent_fragment_downloads": 4,
            "quiet": True,
            "noprogress": True,
            "progress_hooks": [hook],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not out.exists():
            raise RuntimeError(f"Expected {out} but it was not produced.")
        self.log.emit(f"[{item_index}/{total_items}] ok    {label}")


def _video_id_from_url(url: str) -> str:
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(url).query)
    return (qs.get("v") or ["video"])[0]
