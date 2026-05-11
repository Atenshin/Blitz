"""yt-dlp wrappers: fetch playlist metadata and download single videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp


@dataclass
class PlaylistEntry:
    video_id: str
    title: str
    url: str


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
) -> Path:
    """Download a single video to out_path (which should end in .mp4).

    If out_path already exists, this is a no-op. The file is normalized to mp4
    via yt-dlp's merge_output_format.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    # yt-dlp uses outtmpl without the extension to allow remuxing.
    out_no_ext = out_path.with_suffix("")
    ydl_opts: dict[str, Any] = {
        "outtmpl": str(out_no_ext) + ".%(ext)s",
        "format": format_spec,
        "merge_output_format": "mp4",
        "retries": retries,
        "concurrent_fragment_downloads": 4,
        "quiet": False,
        "noprogress": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    if not out_path.exists():
        # yt-dlp may have produced .mkv if merge failed; surface the actual file.
        candidates = list(out_path.parent.glob(out_path.stem + ".*"))
        if candidates:
            raise RuntimeError(
                f"Expected {out_path}, got {candidates[0]}. "
                f"Install ffmpeg or change container in config."
            )
        raise RuntimeError(f"Download finished but {out_path} not found.")
    return out_path
