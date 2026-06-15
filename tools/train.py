"""Train a YOLOv8 model on the labeled FRC dataset.

Reads `training.*` from configs/config.yaml so hyperparameter tweaks live in
one place. Run this once after labeling; output goes to models/runs/<name>/
and the best checkpoint is copied to the path in `detection.model_path`.

Typical usage:
    python tools/train.py

Override knobs:
    python tools/train.py --epochs 50 --batch 4
    python tools/train.py --base-model yolov8s.pt   # smaller, faster
    python tools/train.py --base-model yolov8x.pt   # biggest, may OOM at 1280

If you hit CUDA out-of-memory:
    1. Lower --batch (try 4, then 2)
    2. Or lower --imgsz (try 960, then 640)
    3. Or use a smaller base model
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    cfg = load_cfg()
    t = cfg.get("training", {})
    aug = t.get("augmentation", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path,
                   default=REPO_ROOT / t.get("dataset_yaml", ""),
                   help="Path to data.yaml (or data_split.yaml).")
    p.add_argument("--base-model", default=t.get("base_model", "yolov8m.pt"),
                   help="Pretrained model to start from. yolov8n/s/m/l/x.")
    p.add_argument("--imgsz", type=int, default=t.get("imgsz", 1280))
    p.add_argument("--batch", type=int, default=t.get("batch", 8))
    p.add_argument("--epochs", type=int, default=t.get("epochs", 100))
    p.add_argument("--patience", type=int, default=t.get("patience", 20))
    p.add_argument("--workers", type=int, default=t.get("workers", 4))
    p.add_argument("--cache", default=t.get("cache", "ram"),
                   help='"ram" / "disk" / false. RAM cache 3-5x speedup after '
                        'epoch 1 but needs ~3 GB free RAM at imgsz 960.')
    p.add_argument("--device", default=cfg.get("detection", {}).get("device", 0),
                   help='CUDA device index (0, 1, ...) or "cpu".')
    p.add_argument("--project", type=Path,
                   default=REPO_ROOT / t.get("project", "models/runs"))
    p.add_argument("--name", default=None,
                   help="Run subfolder name. Default: yolov8m_<timestamp>.")
    p.add_argument("--resume", action="store_true",
                   help="Resume the most recent run in --project.")
    args = p.parse_args(argv)

    if not args.data.exists():
        print(f"Dataset YAML not found: {args.data}", file=sys.stderr)
        print(f"Did you run tools/resplit_dataset.py yet?", file=sys.stderr)
        return 2

    # Late import — pulls in torch + cv2 and surfacing those errors here is
    # nicer than at module load time when the user just wants --help.
    from ultralytics import YOLO

    if args.name is None:
        stem = Path(args.base_model).stem
        args.name = f"{stem}_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"[train] data={args.data.relative_to(REPO_ROOT)}")
    print(f"[train] base={args.base_model} imgsz={args.imgsz} batch={args.batch} "
          f"epochs={args.epochs} patience={args.patience} device={args.device}")
    print(f"[train] run name: {args.name}")
    print(f"[train] output:   {args.project / args.name}")

    # Normalize the cache flag — Ultralytics wants True/False/"ram"/"disk".
    cache = args.cache
    if isinstance(cache, str):
        if cache.lower() in ("false", "no", "off", ""):
            cache = False
        elif cache.lower() in ("true", "yes", "on"):
            cache = True

    model = YOLO(args.base_model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        workers=args.workers,
        cache=cache,
        device=args.device,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        # Augmentation overrides (defaults dialed down for Roboflow-pre-augmented data).
        mosaic=aug.get("mosaic", 0.5),
        mixup=aug.get("mixup", 0.1),
        hsv_h=aug.get("hsv_h", 0.015),
        hsv_s=aug.get("hsv_s", 0.4),
        hsv_v=aug.get("hsv_v", 0.3),
        degrees=aug.get("degrees", 2.0),
        translate=aug.get("translate", 0.05),
        scale=aug.get("scale", 0.3),
        fliplr=aug.get("fliplr", 0.5),
        flipud=aug.get("flipud", 0.0),
        # Plot training curves to disk for later inspection.
        plots=True,
        save=True,
    )

    # Copy the best weights to the canonical inference path so M3.4 can find
    # them without knowing the run name.
    run_dir = args.project / args.name
    best_pt = run_dir / "weights" / "best.pt"
    target = REPO_ROOT / cfg.get("detection", {}).get(
        "model_path", "models/frc-2026/best.pt"
    )
    if best_pt.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, target)
        size_mb = target.stat().st_size / (1024 * 1024)
        print(f"\n[done] Best weights: {best_pt.relative_to(REPO_ROOT)}")
        print(f"[done] Copied to:    {target.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
        print(f"\nNext: run inference with `python tools/run_inference.py` (lands in M3.4).")
    else:
        print(f"\n[warn] Training finished but {best_pt} doesn't exist?",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
