"""FRC Match Analysis — Milestone 1 CLI.

Three modes:

  python main.py --event 2025miket
      Use TBA's own video links for the event. No playlist URL needed.

  python main.py --playlist <URL> --event 2025miket
      Walk a YouTube playlist, match each title against TBA's match list for
      the given event, download in order.

  python main.py --youtube <URL> [--match 2025miket_qm12]
      Download a single video. If --match is given, store it under that key.

Common flags:
  --dry-run       Print what would happen, don't download.
  --limit N       Only download the first N matched videos.
  --config PATH   Override configs/config.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from api.tba import Match, TBAClient, load_auth_key
from downloader.matcher import parse_title
from downloader.youtube import (
    download_video,
    fetch_playlist_entries,
    fetch_video_title,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"
DEFAULT_SECRETS = REPO_ROOT / "configs" / "secrets.yaml"


# ---------------------------------------------------------------------------
# Config


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def video_path_for(cfg: dict, match_key: str) -> Path:
    event_key = match_key.split("_", 1)[0]
    return REPO_ROOT / cfg["paths"]["videos"] / event_key / f"{match_key}.mp4"


# ---------------------------------------------------------------------------
# Modes


def run_event_mode(cfg: dict, tba: TBAClient, event_key: str, limit: int | None, dry_run: bool) -> int:
    """Download every match for an event using TBA's own video links."""
    cache_root = REPO_ROOT / cfg["paths"]["videos"]
    tba.cache_dir = cache_root
    matches = tba.event_matches(event_key)
    print(f"[tba] {len(matches)} matches for {event_key}")

    targets: list[tuple[Match, str]] = []  # (match, youtube_url)
    for m in matches:
        yt = next((v for v in m.videos if v.type == "youtube" and v.key), None)
        if yt is None:
            continue
        targets.append((m, yt.youtube_url))  # type: ignore[arg-type]

    print(f"[tba] {len(targets)} matches have YouTube videos linked on TBA.")
    if limit:
        targets = targets[:limit]

    return _download_targets(cfg, targets, dry_run)


def run_playlist_mode(
    cfg: dict, tba: TBAClient, playlist_url: str, event_key: str, limit: int | None, dry_run: bool
) -> int:
    """Walk a playlist, parse each title, map to a TBA match key, download."""
    cache_root = REPO_ROOT / cfg["paths"]["videos"]
    tba.cache_dir = cache_root
    matches = tba.event_matches(event_key)
    matches_by_key = {m.key: m for m in matches}
    print(f"[tba] {len(matches)} matches loaded for {event_key}")

    entries = fetch_playlist_entries(playlist_url)
    print(f"[yt]  {len(entries)} videos in playlist")

    targets: list[tuple[Match, str]] = []
    unmatched: list[tuple[str, str]] = []
    for entry in entries:
        parsed = parse_title(entry.title)
        if parsed is None:
            unmatched.append((entry.video_id, entry.title))
            continue
        key = parsed.key(event_key)
        match = matches_by_key.get(key)
        if match is None:
            unmatched.append((entry.video_id, entry.title))
            continue
        targets.append((match, entry.url))

    print(f"[map] {len(targets)} videos auto-mapped, {len(unmatched)} unmatched")
    if unmatched:
        unmatched_log = cache_root / event_key / "unmatched.json"
        unmatched_log.parent.mkdir(parents=True, exist_ok=True)
        unmatched_log.write_text(
            json.dumps(
                [{"video_id": v, "title": t} for v, t in unmatched], indent=2
            ),
            encoding="utf-8",
        )
        print(f"[map] wrote {unmatched_log} (manual mapping UI lands in a later milestone)")

    if limit:
        targets = targets[:limit]
    return _download_targets(cfg, targets, dry_run)


def run_single_mode(
    cfg: dict, video_url: str, match_key: str | None, dry_run: bool
) -> int:
    if match_key:
        out = video_path_for(cfg, match_key)
        label = match_key
    else:
        # No match association → drop into videos/_unassigned/<id>.mp4 using the
        # YouTube video id as the filename. The user can rename + map later.
        title = fetch_video_title(video_url)
        # Try to extract a usable id from the URL via yt-dlp's noplaylist path.
        # We already have the title — derive the id from the URL.
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(video_url).query)
        vid = (qs.get("v") or [""])[0] or "video"
        out = REPO_ROOT / cfg["paths"]["videos"] / "_unassigned" / f"{vid}.mp4"
        label = f"_unassigned/{vid}  (title: {title})"

    print(f"[dl ] {label}")
    if dry_run:
        return 0
    download_video(
        video_url,
        out,
        format_spec=cfg["download"]["format"],
        retries=cfg["download"]["retries"],
    )
    print(f"[ok ] {out}")
    return 0


# ---------------------------------------------------------------------------
# Shared download loop


def _download_targets(
    cfg: dict, targets: list[tuple[Match, str]], dry_run: bool
) -> int:
    fmt = cfg["download"]["format"]
    retries = cfg["download"]["retries"]
    failed: list[tuple[str, str]] = []

    for i, (match, url) in enumerate(targets, 1):
        out = video_path_for(cfg, match.key)
        prefix = f"[{i}/{len(targets)}]"
        if out.exists():
            print(f"{prefix} skip  {match.key}  (exists)")
            continue
        print(f"{prefix} dl    {match.key}  <- {url}")
        if dry_run:
            continue
        try:
            download_video(url, out, format_spec=fmt, retries=retries)
        except Exception as e:  # yt-dlp surfaces many exception types
            print(f"{prefix} FAIL  {match.key}: {e}")
            failed.append((match.key, str(e)))

    if failed:
        print(f"\n{len(failed)} downloads failed:")
        for k, err in failed:
            print(f"  {k}: {err}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="FRC Match Analyzer")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--gui", action="store_true", help="Launch the GUI (default if no other mode flag).")
    mode.add_argument("--event", help="TBA event key, e.g. 2025miket. Uses TBA's own video links.")
    mode.add_argument("--playlist", help="YouTube playlist URL. Requires --event-key for matching.")
    mode.add_argument("--youtube", help="Single YouTube video URL.")

    p.add_argument("--event-key", help="(With --playlist) TBA event to match titles against.")
    p.add_argument("--match", help="(With --youtube) TBA match key to store the video under.")
    p.add_argument("--limit", type=int, default=None, help="Cap number of downloads.")
    p.add_argument("--dry-run", action="store_true", help="Print plan, do not download.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--secrets", type=Path, default=DEFAULT_SECRETS)

    args = p.parse_args(argv)
    cfg = load_config(args.config)

    # Default to GUI when no CLI download mode is specified.
    no_cli_mode = not (args.event or args.playlist or args.youtube)
    if args.gui or no_cli_mode:
        from gui.app import run as run_gui
        try:
            auth = load_auth_key(args.secrets)
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] {e}\n[warn] TBA features in the GUI will be disabled.", file=sys.stderr)
            auth = None
        return run_gui(REPO_ROOT, cfg, auth)

    if args.youtube:
        return run_single_mode(cfg, args.youtube, args.match, args.dry_run)

    # event/playlist modes need TBA
    auth = load_auth_key(args.secrets)
    tba = TBAClient(auth)

    if args.event:
        return run_event_mode(cfg, tba, args.event, args.limit, args.dry_run)

    # playlist mode
    event_key = args.event_key or cfg.get("event_key")
    if not event_key:
        print("--playlist requires --event-key (or event_key in config.yaml)", file=sys.stderr)
        return 2
    return run_playlist_mode(cfg, tba, args.playlist, event_key, args.limit, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
