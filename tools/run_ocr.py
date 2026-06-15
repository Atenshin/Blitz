"""Run OCR on tracked robots in a detection cache to attribute team numbers.

Pipeline:
    1. Load the detection cache (must have tracking IDs — run with --tracker
       bytetrack in tools/run_inference.py).
    2. Load the match's TBA metadata (videos/<event>/matches.json) to get
       the 6 team numbers — 3 red + 3 blue.
    3. For each tracked robot, evenly sample up to N frames where it appears,
       open the source video, crop the robot, OCR for digits.
    4. Vote across readings per track_id, picking the alliance-roster-matching
       team number with the highest weighted confidence.
    5. Write identities/<event>/<match_key>.json.

Usage:
    python tools/run_ocr.py videos/2026cmptx/2026cmptx_sf1m1.mp4
    python tools/run_ocr.py videos/2026cmptx               # all matches in an event
    python tools/run_ocr.py videos/                        # everything cached
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from detection.identity import (
    IDENTITY_SCHEMA_VERSION,
    MatchIdentities,
    attribute_identity,
)
from detection.inference import cache_path_for
from detection.ocr import BumperReader, crop_robot, team_number_from_team_key
from detection.schema import MatchDetectionCache


def load_cfg() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")) or {}


def resolve_videos(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() == ".mp4" else []
    if not target.is_dir():
        return []
    direct = sorted(target.glob("*.mp4"))
    if direct:
        return direct
    return sorted(target.glob("*/*.mp4"))


def load_match_alliances(video: Path, videos_root: Path) -> tuple[list[str], list[str]] | None:
    """Return (red_teams, blue_teams) for the match this video represents,
    or None if no TBA metadata is cached."""
    event_dir = video.parent
    matches_json = event_dir / "matches.json"
    if not matches_json.exists():
        return None
    data = json.loads(matches_json.read_text(encoding="utf-8"))
    match_key = video.stem
    for m in data:
        if m["key"] == match_key:
            return m.get("red_teams") or [], m.get("blue_teams") or []
    return None


def process_one(
    video: Path,
    cache_path: Path,
    out_path: Path,
    videos_root: Path,
    reader: BumperReader,
    samples_per_track: int,
    min_track_appearances: int,
) -> tuple[int, int, int]:
    """OCR every qualifying tracked robot in one match. Returns
    (tracks_attributed, tracks_total, samples_taken)."""
    cache = MatchDetectionCache.read(cache_path)

    if not cache.tracking_used:
        print(f"  cache has no tracking IDs — re-run tools/run_inference.py with --tracker bytetrack",
              file=sys.stderr)
        return (0, 0, 0)

    # Group detections by track ID, keeping only robots.
    track_frames: dict[int, list[tuple[int, list[float], str, float]]] = defaultdict(list)
    track_class_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for f in cache.frames:
        for d in f.detections:
            if d.object_id is None:
                continue
            if d.name not in ("robot_blue", "robot_red"):
                continue
            track_frames[d.object_id].append((f.frame_idx, d.bbox, d.name, d.conf))
            track_class_counts[d.object_id][d.name] += 1

    # Apply min_track_appearances — skip very brief tracks (likely false positives)
    eligible_tracks = {
        tid: frames for tid, frames in track_frames.items()
        if len(frames) >= min_track_appearances
    }
    print(f"  {len(track_frames)} robot tracks, {len(eligible_tracks)} eligible "
          f"(≥{min_track_appearances} appearances)")

    # Alliance rosters
    alliances = load_match_alliances(video, videos_root)
    if alliances is None:
        print(f"  no TBA metadata for {video.stem}, can't validate readings",
              file=sys.stderr)
        return (0, len(eligible_tracks), 0)
    red_teams, blue_teams = alliances
    red_numeric = {team_number_from_team_key(t) for t in red_teams}
    blue_numeric = {team_number_from_team_key(t) for t in blue_teams}

    # Plan upfront: which (frame_idx, track_id, bbox) tuples do we need to OCR?
    # Building this list before touching the video means we can walk the video
    # linearly with cv2.grab()/retrieve() and decode each frame at most once —
    # 100x faster than seeking with cap.set(POS_FRAMES, X) for each sample on
    # Windows + OpenCV.
    needs_ocr: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for tid, frames in eligible_tracks.items():
        step = max(1, len(frames) // samples_per_track)
        sampled = frames[::step][:samples_per_track]
        for frame_idx, bbox, _name, _conf in sampled:
            needs_ocr[frame_idx].append((tid, bbox))

    sorted_target_frames = sorted(needs_ocr.keys())
    if not sorted_target_frames:
        print(f"  no tracks eligible for OCR", file=sys.stderr)
        return (0, 0, 0)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  could not open {video}", file=sys.stderr)
        return (0, 0, 0)

    readings_by_track: dict[int, list[tuple[str, float]]] = defaultdict(list)
    samples_total = 0

    # Linear walk: grab() advances past every frame without decoding;
    # retrieve() decodes only when we hit a sample point.
    target_iter = iter(sorted_target_frames)
    next_target = next(target_iter, None)
    frame_idx = 0
    pbar = tqdm(
        total=len(sorted_target_frames),
        unit="sample",
        desc="OCR",
        leave=False,
    )
    while next_target is not None:
        if frame_idx == next_target:
            ok, frame = cap.retrieve() if cap.grab() else (False, None)
            if ok and frame is not None:
                for tid, bbox in needs_ocr[frame_idx]:
                    crop = crop_robot(frame, bbox)
                    for text, ocr_conf in reader.read_digits(crop):
                        readings_by_track[tid].append((text, ocr_conf))
                    samples_total += 1
                    pbar.update(1)
            next_target = next(target_iter, None)
        else:
            # Cheap path: skip past frames we don't need (no decode).
            if not cap.grab():
                break
        frame_idx += 1
    pbar.close()
    cap.release()

    # Aggregate readings per track into a final identity.
    identities = MatchIdentities(
        schema_version=IDENTITY_SCHEMA_VERSION,
        match_key=cache.match_key,
        red_teams=red_teams,
        blue_teams=blue_teams,
        samples_taken=0,
    )
    for tid in eligible_tracks.keys():
        identities.tracks[tid] = attribute_identity(
            track_id=tid,
            class_counts=track_class_counts[tid],
            raw_readings=readings_by_track[tid],
            red_teams_numeric=red_numeric,
            blue_teams_numeric=blue_numeric,
        )
    # The pre-existing post-loop block below handles file writing.
    identities.samples_taken = samples_total
    identities.write(out_path)

    n_attributed = sum(
        1 for t in identities.tracks.values()
        if t.team_number is not None
    )
    return (n_attributed, len(eligible_tracks), samples_total)


def main(argv: list[str] | None = None) -> int:
    cfg = load_cfg()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", default=str(REPO_ROOT / "videos"))
    p.add_argument("--detections", type=Path,
                   default=REPO_ROOT / cfg.get("paths", {}).get("detections", "detections"))
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "identities")
    p.add_argument("--videos-root", type=Path,
                   default=REPO_ROOT / cfg.get("paths", {}).get("videos", "videos"))
    p.add_argument("--samples-per-track", type=int, default=25,
                   help="Max OCR samples per tracked robot. Default 25 — enough "
                        "to handle robots that face the camera briefly while "
                        "keeping wall time manageable.")
    p.add_argument("--min-track-appearances", type=int, default=20,
                   help="Skip tracks with fewer than this many detections "
                        "(usually false positives). Default 20.")
    p.add_argument("--cpu", action="store_true", help="Force CPU-mode OCR.")
    p.add_argument("--force", action="store_true",
                   help="Re-OCR even if identity file exists.")
    args = p.parse_args(argv)

    target = Path(args.target).resolve()
    videos = resolve_videos(target)
    if not videos:
        print(f"No videos found under {target}", file=sys.stderr)
        return 2

    print(f"Initializing EasyOCR ({'CPU' if args.cpu else 'GPU'})…")
    t0 = time.time()
    reader = BumperReader(gpu=not args.cpu)
    reader._ensure_loaded()
    print(f"  ready in {time.time() - t0:.1f}s")
    print(f"  {len(videos)} videos to process")

    started_all = time.time()
    grand_total = grand_attr = 0
    for i, video in enumerate(videos, 1):
        cache_path = cache_path_for(video, args.detections)
        if not cache_path.exists():
            print(f"[{i}/{len(videos)}] skip  {video.stem}  (no detection cache — "
                  f"run tools/run_inference.py first)")
            continue
        event_key = video.parent.name
        out_path = args.out / event_key / f"{video.stem}.json"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(videos)}] skip  {video.stem}  (identity file exists)")
            continue

        print(f"[{i}/{len(videos)}] run   {video.stem}")
        t = time.time()
        attr, total, samples = process_one(
            video=video,
            cache_path=cache_path,
            out_path=out_path,
            videos_root=args.videos_root,
            reader=reader,
            samples_per_track=args.samples_per_track,
            min_track_appearances=args.min_track_appearances,
        )
        elapsed = time.time() - t
        grand_total += total
        grand_attr += attr
        print(f"[{i}/{len(videos)}] ok    {video.stem}  "
              f"{attr}/{total} tracks attributed, {samples} OCR samples, {elapsed:.1f}s")

    total_elapsed = time.time() - started_all
    print(f"\n[done] {grand_attr}/{grand_total} tracks attributed in {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
