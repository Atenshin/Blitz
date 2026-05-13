"""Extract diverse frames from cached match videos for labeling.

Reads every mp4 under videos/<event>/, samples frames at a configurable rate,
and writes deduplicated JPGs to datasets/raw/<event>/. A perceptual hash
collapses near-identical frames (intros, scoreboard screens, slow camera
shots), so the resulting set is something a human can usefully label without
spending hours rejecting duplicates.

Usage
-----

  # Default: all videos under videos/, 1 fps sampling, dedupe enabled
  python tools/extract_frames.py

  # One specific event
  python tools/extract_frames.py videos/2025cmptx

  # One specific match, denser sampling, no dedupe
  python tools/extract_frames.py videos/2025cmptx/2025cmptx_sf1m1.mp4 --fps 2 --no-dedupe

Output
------

  datasets/raw/<event>/<match_key>__<sec>.jpg     extracted frames
  datasets/raw/<event>/_manifest.json             frame -> source mapping

The manifest is what you upload alongside images to Roboflow — it preserves
the link back to the source video and timestamp so you can re-derive frames
later (e.g. for active-learning rounds).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import imagehash
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FrameRecord:
    file: str            # filename only, relative to the per-event dir
    sec: float
    frame_idx: int
    phash: str


@dataclass
class VideoRecord:
    video: str           # relative path from repo root
    match_key: str
    src_fps: float
    duration_sec: float
    frames_extracted: int
    frames_kept: int
    frames: list[FrameRecord] = field(default_factory=list)


@dataclass
class Manifest:
    event_key: str
    fps_sampled: float
    hamming_threshold: int
    resize_max: int | None
    jpeg_quality: int
    videos: list[VideoRecord] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_key": self.event_key,
                "fps_sampled": self.fps_sampled,
                "hamming_threshold": self.hamming_threshold,
                "resize_max": self.resize_max,
                "jpeg_quality": self.jpeg_quality,
                "videos": [
                    {**asdict(v), "frames": [asdict(f) for f in v.frames]}
                    for v in self.videos
                ],
            },
            indent=2,
        )


def resolve_videos(target: Path) -> list[Path]:
    """Resolve the input arg into a flat list of mp4 paths."""
    if target.is_file():
        return [target] if target.suffix.lower() == ".mp4" else []
    if not target.is_dir():
        return []
    # If the dir itself looks like an event dir (contains mp4s directly), use those.
    direct = sorted(target.glob("*.mp4"))
    if direct:
        return direct
    # Otherwise, dive one level (e.g. videos/ → videos/<event>/*.mp4).
    return sorted(target.glob("*/*.mp4"))


def event_key_for(video: Path) -> str:
    """Take 2025cmptx out of videos/2025cmptx/2025cmptx_sf1m1.mp4."""
    name = video.stem
    if "_" in name:
        return name.split("_", 1)[0]
    return video.parent.name


def resize_keep_aspect(frame, max_dim: int):
    """Downscale a BGR frame so its longest side == max_dim (no upscale)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def extract_one(
    video_path: Path,
    out_dir: Path,
    fps_target: float,
    hamming_threshold: int,
    resize_max: int | None,
    jpeg_quality: int,
    kept_hashes_global: list,
    dedupe: bool,
) -> VideoRecord:
    """Pull frames out of one video, dedupe inline, write the survivors."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps else 0.0

    step = max(1, int(round(src_fps / fps_target)))
    match_key = video_path.stem
    record = VideoRecord(
        video=video_path.relative_to(REPO_ROOT).as_posix(),
        match_key=match_key,
        src_fps=src_fps,
        duration_sec=duration,
        frames_extracted=0,
        frames_kept=0,
    )

    # Progress bar counts sampled frames, not total decoder frames — the
    # grab()-only path skips decoding so total frames isn't a useful unit.
    estimated_samples = total_frames // step + 1
    pbar = tqdm(total=estimated_samples, desc=match_key, unit="f", leave=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        # Fast path: grab() pulls the next frame from the stream without
        # decoding it. We only call retrieve() (which actually decodes) when
        # frame_idx aligns with our sample step.
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue
            record.frames_extracted += 1
            pbar.update(1)

            if resize_max is not None:
                frame = resize_keep_aspect(frame, resize_max)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h = imagehash.phash(pil, hash_size=8)

            keep = True
            if dedupe:
                # Hamming distance against every kept hash so far — both this
                # video and earlier videos in the same batch. 5–7 catches
                # near-duplicates without false positives on game action.
                for existing in kept_hashes_global:
                    if h - existing <= hamming_threshold:
                        keep = False
                        break

            if keep:
                sec = frame_idx / src_fps
                fname = f"{match_key}__{int(sec):05d}.jpg"
                out_path = out_dir / fname
                cv2.imwrite(str(out_path), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                record.frames.append(FrameRecord(
                    file=fname, sec=round(sec, 2),
                    frame_idx=frame_idx, phash=str(h),
                ))
                kept_hashes_global.append(h)
                record.frames_kept += 1

        frame_idx += 1

    pbar.close()
    cap.release()
    return record


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract diverse frames for labeling.")
    p.add_argument(
        "target",
        nargs="?",
        default=str(REPO_ROOT / "videos"),
        help="Path: an mp4 file, an event dir, or the videos/ root. "
             "Default: videos/ (every event).",
    )
    p.add_argument("--fps", type=float, default=1.0,
                   help="Sampling rate in frames per second. Default 1.")
    p.add_argument("--hamming-threshold", type=int, default=6,
                   help="Max phash distance to treat two frames as duplicates. "
                        "Lower = stricter (more frames kept). Default 6.")
    p.add_argument("--no-dedupe", action="store_true",
                   help="Disable perceptual-hash dedupe. Keeps every sampled frame.")
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "datasets" / "raw",
                   help="Output root. Per-event subdirs are created beneath it.")
    p.add_argument("--resize", type=int, default=1280,
                   help="Downscale frames so the longest side is at most this many "
                        "pixels. 0 disables. Default 1280 (saves disk + speeds labeling).")
    p.add_argument("--quality", type=int, default=88,
                   help="JPEG quality 1-100. Default 88.")
    args = p.parse_args(argv)

    target = Path(args.target).resolve()
    videos = resolve_videos(target)
    if not videos:
        print(f"No mp4 files found under {target}", file=sys.stderr)
        return 2

    resize_max = args.resize if args.resize > 0 else None

    # Group by event so each event gets its own manifest and output dir.
    by_event: dict[str, list[Path]] = {}
    for v in videos:
        by_event.setdefault(event_key_for(v), []).append(v)

    print(f"Found {len(videos)} videos across {len(by_event)} event(s).")
    print(f"Sampling {args.fps} fps, dedupe={'off' if args.no_dedupe else f'on (threshold {args.hamming_threshold})'}, "
          f"resize_max={resize_max}, quality={args.quality}")

    grand_extracted = 0
    grand_kept = 0
    started = time.time()

    for event_key, event_videos in by_event.items():
        out_dir = args.out / event_key
        manifest = Manifest(
            event_key=event_key,
            fps_sampled=args.fps,
            hamming_threshold=args.hamming_threshold,
            resize_max=resize_max,
            jpeg_quality=args.quality,
        )
        kept_hashes: list = []

        print(f"\n[event] {event_key}: {len(event_videos)} videos")
        for video in event_videos:
            record = extract_one(
                video, out_dir,
                fps_target=args.fps,
                hamming_threshold=args.hamming_threshold,
                resize_max=resize_max,
                jpeg_quality=args.quality,
                kept_hashes_global=kept_hashes,
                dedupe=not args.no_dedupe,
            )
            manifest.videos.append(record)
            grand_extracted += record.frames_extracted
            grand_kept += record.frames_kept
            print(f"  {record.match_key}: "
                  f"{record.frames_extracted} sampled -> {record.frames_kept} kept")

        manifest_path = out_dir / "_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        print(f"  [manifest] {manifest_path}")

    elapsed = time.time() - started
    drop_pct = (1 - grand_kept / grand_extracted) * 100 if grand_extracted else 0
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"{grand_extracted} sampled -> {grand_kept} kept "
        f"({drop_pct:.0f}% deduped)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
