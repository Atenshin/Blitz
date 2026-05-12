"""Run YOLO over cached match videos and write per-match detection caches.

After training (`python tools/train.py`) writes weights to
`models/frc-2026/best.pt`, this script reads those weights and processes
every .mp4 under videos/, producing one JSON file per match at
`detections/<event>/<match_key>.json`.

Usage:
    python tools/run_inference.py                           # all videos
    python tools/run_inference.py videos/2026cmptx          # one event
    python tools/run_inference.py videos/2026cmptx/2026cmptx_sf1m1.mp4

Flags:
    --inference-fps 15      Run YOLO N times per source second (default 15)
    --confidence 0.35       Score threshold (default from config)
    --iou 0.5               NMS IoU (default from config)
    --imgsz 1280            Inference image size (default from config)
    --device 0              CUDA device or "cpu"
    --force                 Re-run even if cache exists
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from detection.inference import FrameDetector, cache_path_for, process_video


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


def main(argv: list[str] | None = None) -> int:
    cfg = load_cfg()
    d = cfg.get("detection", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", nargs="?", default=str(REPO_ROOT / "videos"),
                   help="Video file, event dir, or videos/ root. Default: videos/")
    p.add_argument("--model", type=Path,
                   default=REPO_ROOT / d.get("model_path", "models/frc-2026/best.pt"))
    p.add_argument("--inference-fps", type=float, default=15.0,
                   help="How many frames per source second to run YOLO on. Default 15.")
    p.add_argument("--confidence", type=float, default=d.get("confidence", 0.35))
    p.add_argument("--iou", type=float, default=d.get("iou", 0.5))
    p.add_argument("--imgsz", type=int, default=d.get("imgsz", 1280))
    p.add_argument("--device", default=d.get("device", 0),
                   help='CUDA index (0, 1, ...) or "cpu".')
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / cfg.get("paths", {}).get("detections", "detections"))
    p.add_argument("--force", action="store_true",
                   help="Re-run detection even if a cache file already exists.")
    args = p.parse_args(argv)

    target = Path(args.target).resolve()
    videos = resolve_videos(target)
    if not videos:
        print(f"No mp4 files found under {target}", file=sys.stderr)
        return 2

    # FrameDetector handles both real paths and bare Ultralytics model names
    # (e.g. "yolov8m.pt"), so don't pre-validate here — let it raise with the
    # right message.

    try:
        model_label = args.model.relative_to(REPO_ROOT)
    except ValueError:
        # Bare Ultralytics model name (e.g. "yolov8m.pt") — not in repo root.
        model_label = args.model
    print(f"[init] loading {model_label}")
    t0 = time.time()
    detector = FrameDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )
    print(f"[init] ready in {time.time() - t0:.1f}s — classes: {detector.class_names}")
    print(f"[init] {len(videos)} videos to process, inference @ {args.inference_fps} Hz")

    skipped = 0
    processed = 0
    failed: list[tuple[str, str]] = []
    started_all = time.time()

    for i, video in enumerate(videos, 1):
        out_path = cache_path_for(video, args.out)
        prefix = f"[{i}/{len(videos)}]"
        if out_path.exists() and not args.force:
            print(f"{prefix} skip  {video.stem}  (cache exists: {out_path.relative_to(REPO_ROOT)})")
            skipped += 1
            continue

        print(f"{prefix} run   {video.stem}")
        t = time.time()
        # Single-video progress bar measures seconds of source video processed.
        pbar = tqdm(total=0, desc=video.stem, unit="s", bar_format="{desc}: {n:.0f}s/{total:.0f}s [{elapsed}<{remaining}]", leave=False)

        def on_progress(sec: float, total: float) -> None:
            if pbar.total != int(total):
                pbar.total = int(total)
            pbar.n = sec
            pbar.refresh()

        try:
            cache = process_video(
                video_path=video,
                detector=detector,
                inference_fps=args.inference_fps,
                repo_root=REPO_ROOT,
                progress_cb=on_progress,
            )
            cache.write(out_path)
            pbar.close()
            elapsed = time.time() - t
            n_dets = sum(len(f.detections) for f in cache.frames)
            print(f"{prefix} ok    {video.stem}  "
                  f"({len(cache.frames)} frames, {n_dets} detections, {elapsed:.1f}s)")
            processed += 1
        except Exception as e:
            pbar.close()
            print(f"{prefix} FAIL  {video.stem}: {e}", file=sys.stderr)
            failed.append((video.stem, str(e)))

    total_elapsed = time.time() - started_all
    print(f"\n[done] {processed} processed, {skipped} skipped, {len(failed)} failed "
          f"in {total_elapsed:.1f}s.")
    if failed:
        for k, err in failed:
            print(f"  {k}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
