"""yt-dlp wrappers: fetch playlist metadata and download single videos."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp


@dataclass
class PlaylistEntry:
    video_id: str
    title: str
    url: str


class UploaderRejected(Exception):
    """Raised when a video's uploader doesn't match the allowed list."""

    def __init__(self, url: str, uploader: str):
        self.url = url
        self.uploader = uploader
        super().__init__(f"uploader '{uploader}' not in allowed list ({url})")


def probe_uploader(video_url: str) -> tuple[str, str]:
    """Return (uploader_signature, title) for a single video without downloading.

    The signature is a "|"-joined string of every uploader-ish field yt-dlp
    surfaced (uploader display name, uploader_id handle, channel display name).
    Checking against this string with `uploader_is_allowed` catches all the
    naming variants YouTube exposes for the same channel.
    """
    ydl_opts = {"quiet": True, "skip_download": True, "noprogress": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False) or {}
    fields = [
        info.get("uploader") or "",
        info.get("uploader_id") or "",
        info.get("channel") or "",
        info.get("channel_id") or "",
    ]
    signature = " | ".join(f for f in fields if f)
    title = info.get("title") or ""
    return signature, title


def uploader_is_allowed(uploader_signature: str, allowed: list[str]) -> bool:
    """Case-insensitive substring match against any of the uploader fields.

    Empty allowed list means allow all.
    """
    if not allowed:
        return True
    sig = uploader_signature.lower()
    return any(a.lower() in sig for a in allowed)


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Make a string safe for use as a filename on Windows."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")
    return cleaned[:max_len] or "video"


def fetch_playlist_entries(playlist_url: str) -> list[PlaylistEntry]:
    """Return [video_id, title, url] for each item in a playlist without downloading."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    entries = info.get("entries") or []
    out: list[PlaylistEntry] = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id") or ""
        if not vid:
            continue
        out.append(
            PlaylistEntry(
                video_id=vid,
                title=e.get("title") or "",
                url=e.get("url") or f"https://www.youtube.com/watch?v={vid}",
            )
        )
    return out


def fetch_video_title(video_url: str) -> str:
    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info.get("title") or ""


def download_video(
    video_url: str,
    out_path: Path,
    format_spec: str = "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    retries: int = 3,
    allowed_uploaders: list[str] | None = None,
    progress_hook=None,
) -> Path:
    """Download a single video to out_path (which should end in .mp4).

    If out_path already exists, this is a no-op. The file is normalized to mp4
    via yt-dlp's merge_output_format.

    If allowed_uploaders is provided, the video's uploader is probed first and
    UploaderRejected is raised when there's no match — no bytes are downloaded.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    if allowed_uploaders:
        uploader, _ = probe_uploader(video_url)
        if not uploader_is_allowed(uploader, allowed_uploaders):
            raise UploaderRejected(video_url, uploader)

    out_no_ext = out_path.with_suffix("")
    ydl_opts: dict[str, Any] = {
        "outtmpl": str(out_no_ext) + ".%(ext)s",
        "format": format_spec,
        "merge_output_format": "mp4",
        "retries": retries,
        "concurrent_fragment_downloads": 4,
        "quiet": progress_hook is not None,
        "noprogress": progress_hook is not None,
    }
    if progress_hook is not None:
        ydl_opts["progress_hooks"] = [progress_hook]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if not out_path.exists():
        candidates = list(out_path.parent.glob(out_path.stem + ".*"))
        if candidates:
            raise RuntimeError(
                f"Expected {out_path}, got {candidates[0]}. "
                f"Install ffmpeg or change container in config."
            )
        raise RuntimeError(f"Download finished but {out_path} not found.")
    return out_path
