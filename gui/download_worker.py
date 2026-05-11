"""QThread-friendly wrapper around the downloader functions.

Three modes — single URL, playlist (event key optional), event-only — all of
which honor an "allowed uploaders" filter so non-FRC fan uploads are rejected
before any bytes are pulled.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

import yt_dlp

from api.tba import Match, TBAClient
from downloader.matcher import parse_title
from downloader.youtube import (
    UploaderRejected,
    fetch_playlist_entries,
    probe_uploader,
    sanitize_filename,
    uploader_is_allowed,
)


class DownloadWorker(QObject):
    """Lives on a QThread. Trigger via run_*() slots; listen on signals."""

    progress = pyqtSignal(str, float)         # (message, percent 0-100; -1 indeterminate)
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int, list)     # (succeeded, failed, error_messages)

    def __init__(
        self,
        videos_root: Path,
        format_spec: str,
        retries: int,
        tba_auth_key: str,
        allowed_uploaders: list[str],
    ):
        super().__init__()
        self.videos_root = videos_root
        self.format_spec = format_spec
        self.retries = retries
        self.allowed_uploaders = allowed_uploaders or []
        self.tba = TBAClient(tba_auth_key, cache_dir=videos_root) if tba_auth_key else None

    # ----- public entry points -----

    def run_single(self, url: str, match_key: str | None) -> None:
        try:
            if match_key:
                out = self._video_path(match_key)
                label = match_key
            else:
                vid = _video_id_from_url(url)
                out = self.videos_root / "_unassigned" / f"{vid}.mp4"
                label = f"_unassigned/{vid}"
            ok = self._download(url, out, label, total_items=1, item_index=1)
            self.finished.emit(1 if ok else 0, 0 if ok else 1, [])
        except Exception as e:
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    def run_playlist(self, playlist_url: str, event_key: str | None) -> None:
        try:
            self.log.emit(f"[yt]  enumerating playlist…")
            entries = fetch_playlist_entries(playlist_url)
            self.log.emit(f"[yt]  {len(entries)} videos found")

            if event_key:
                self._run_playlist_with_event(entries, event_key)
            else:
                self._run_playlist_unassigned(entries, playlist_url)
        except Exception as e:
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    def run_event(self, event_key: str) -> None:
        try:
            if self.tba is None:
                raise RuntimeError("TBA auth key not configured.")
            self.log.emit(f"[tba] fetching matches for {event_key}…")
            matches = self.tba.event_matches(event_key)
            self.log.emit(f"[tba] {len(matches)} matches loaded")
            self._download_matches(matches)
        except Exception as e:
            self.finished.emit(0, 1, [f"{e}\n{traceback.format_exc()}"])

    # ----- internals -----

    def _video_path(self, match_key: str) -> Path:
        event_key = match_key.split("_", 1)[0]
        return self.videos_root / event_key / f"{match_key}.mp4"

    def _run_playlist_with_event(self, entries, event_key: str) -> None:
        if self.tba is None:
            raise RuntimeError("TBA auth key not configured.")
        self.log.emit(f"[tba] fetching matches for {event_key}…")
        matches = self.tba.event_matches(event_key)
        matches_by_key = {m.key: m for m in matches}
        self.log.emit(f"[tba] {len(matches)} matches loaded")

        targets: list[tuple[Path, str, str]] = []  # (out_path, url, label)
        unmatched: list[str] = []
        for entry in entries:
            parsed = parse_title(entry.title)
            if parsed is None:
                unmatched.append(entry.title)
                continue
            key = parsed.key(event_key)
            match = matches_by_key.get(key)
            if match is None:
                unmatched.append(entry.title)
                continue
            targets.append((self._video_path(match.key), entry.url, match.key))

        self.log.emit(f"[map] {len(targets)} mapped, {len(unmatched)} unmatched")
        self._download_targets(targets)

    def _run_playlist_unassigned(self, entries, playlist_url: str) -> None:
        """No event key → drop everything into videos/_unassigned/<title>.mp4."""
        out_dir = self.videos_root / "_unassigned"
        targets: list[tuple[Path, str, str]] = []
        for entry in entries:
            fname = sanitize_filename(entry.title) or entry.video_id
            out = out_dir / f"{fname}.mp4"
            targets.append((out, entry.url, fname))
        self.log.emit(f"[map] no event key — saving {len(targets)} videos to _unassigned/")
        self._download_targets(targets)

    def _download_matches(self, matches: list[Match]) -> None:
        """Event mode: for each match, find the first FRC-uploaded YouTube
        video and download it; skip the match if none qualify."""
        targets: list[tuple[Path, str, str]] = []
        skipped_no_frc: list[str] = []
        skipped_no_video: list[str] = []

        # First pass: pick best video per match (probes uploader). This is
        # ~1s/video but spares us from downloading and then deleting.
        for i, match in enumerate(matches, 1):
            yt_videos = [v for v in match.videos if v.type == "youtube" and v.key]
            if not yt_videos:
                skipped_no_video.append(match.key)
                continue
            picked = None
            for v in yt_videos:
                url = v.youtube_url
                if url is None:
                    continue
                self.progress.emit(f"[scan {i}/{len(matches)}] {match.key}", -1)
                try:
                    uploader, _ = probe_uploader(url)
                except Exception as e:
                    self.log.emit(f"[scan] {match.key}: probe failed ({e})")
                    continue
                if uploader_is_allowed(uploader, self.allowed_uploaders):
                    picked = (self._video_path(match.key), url, match.key)
                    break
                else:
                    self.log.emit(f"[scan] {match.key}: skipping '{uploader}'")
            if picked:
                targets.append(picked)
            else:
                skipped_no_frc.append(match.key)

        self.log.emit(
            f"[scan] {len(targets)} downloadable, "
            f"{len(skipped_no_frc)} skipped (no FRC upload), "
            f"{len(skipped_no_video)} skipped (no video at all)"
        )
        self._download_targets(targets)

    def _download_targets(self, targets: list[tuple[Path, str, str]]) -> None:
        succeeded = 0
        errors: list[str] = []
        for i, (out, url, label) in enumerate(targets, 1):
            if out.exists():
                self.log.emit(f"[{i}/{len(targets)}] skip  {label} (exists)")
                succeeded += 1
                continue
            try:
                ok = self._download(url, out, label, total_items=len(targets), item_index=i)
                if ok:
                    succeeded += 1
            except Exception as e:
                errors.append(f"{label}: {e}")
                self.log.emit(f"[{i}/{len(targets)}] FAIL  {label}: {e}")
        self.finished.emit(succeeded, len(errors), errors)

    def _download(self, url: str, out: Path, label: str, total_items: int, item_index: int) -> bool:
        """Returns True if the file ends up on disk; False if it was skipped."""
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            return True

        # Uploader gate (cheap, no bytes pulled).
        if self.allowed_uploaders:
            try:
                uploader, _ = probe_uploader(url)
            except Exception as e:
                self.log.emit(f"[{item_index}/{total_items}] probe FAIL {label}: {e}")
                raise
            if not uploader_is_allowed(uploader, self.allowed_uploaders):
                self.log.emit(
                    f"[{item_index}/{total_items}] skip  {label} (uploader: '{uploader}')"
                )
                return False

        out_no_ext = out.with_suffix("")

        def hook(d: dict) -> None:
            status = d.get("status")
            if status == "downloading":
                downloaded = d.get("downloaded_bytes") or 0
                total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                pct = (downloaded / total * 100.0) if total else -1.0
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
        return True


def _video_id_from_url(url: str) -> str:
    from urllib.parse import parse_qs, urlparse
    qs = parse_qs(urlparse(url).query)
    return (qs.get("v") or ["video"])[0]
