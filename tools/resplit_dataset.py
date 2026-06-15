"""Re-split a Roboflow-exported YOLOv8 dataset into proper train/valid/test ratios.

Roboflow free tier sometimes ships datasets with near-100% train splits
(e.g. 1640/3/7) — useless for monitoring training. This script doesn't touch
the original files; it writes new file lists + a `data_split.yaml` that
Ultralytics can consume directly.

Key constraint: Roboflow bakes augmentations into the train set as multiple
copies of the same source frame, named like
    <base>_jpg.rf.<32-hex-hash>.jpg
We group by the <base> name so every augmentation of a given source frame
ends up in the SAME split — otherwise an augmentation could leak into
valid/test and silently inflate metrics.

Usage:
    python tools/resplit_dataset.py datasets/labeled/Blitz.v5i.yolov8

Output (alongside the dataset):
    train_split.txt
    val_split.txt
    test_split.txt
    data_split.yaml
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Roboflow's augmentation suffix: "_jpg.rf.<32 hex chars>.jpg"
_RF_SUFFIX = re.compile(r"_jpg\.rf\.[a-f0-9]+\.jpg$", re.IGNORECASE)


def base_name(filename: str) -> str:
    """Strip Roboflow's `_jpg.rf.<hash>.jpg` so all augmentations collapse to
    the same key. Falls back to the stem if the pattern doesn't match."""
    stripped = _RF_SUFFIX.sub("", filename)
    if stripped == filename:
        # No Roboflow suffix — use the bare stem so we still dedupe correctly.
        return Path(filename).stem
    return stripped


def collect_images(dataset_root: Path) -> dict[str, list[Path]]:
    """Walk train/valid/test/images and group every image by its source-frame
    base name. Returns {base_name: [absolute paths to all augmented copies]}."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        if not images_dir.is_dir():
            continue
        for img in images_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            groups[base_name(img.name)].append(img.resolve())
    return groups


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path, help="Path to the unzipped Roboflow YOLOv8 export")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.20)
    p.add_argument("--test", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        print("train + val + test must sum to 1.0", file=sys.stderr)
        return 2

    dataset = args.dataset.resolve()
    if not (dataset / "data.yaml").exists():
        print(f"Not a YOLOv8 dataset (no data.yaml): {dataset}", file=sys.stderr)
        return 2

    print(f"[scan] {dataset}")
    groups = collect_images(dataset)
    total_bases = len(groups)
    total_images = sum(len(v) for v in groups.values())
    print(f"[scan] {total_bases} unique source frames across {total_images} files "
          f"(avg {total_images / total_bases:.1f} augmentations per frame)")

    bases = sorted(groups.keys())
    random.Random(args.seed).shuffle(bases)

    n_train = int(round(len(bases) * args.train))
    n_val = int(round(len(bases) * args.val))
    train_bases = bases[:n_train]
    val_bases = bases[n_train:n_train + n_val]
    test_bases = bases[n_train + n_val:]
    print(f"[split] {len(train_bases)} train / {len(val_bases)} val / {len(test_bases)} test "
          f"(base frames)")

    # For train: include every augmentation. For val/test: keep one file per
    # base (the lexicographically-first one for determinism). The Roboflow
    # augmentations baked into the file are mild — using one of them as a
    # validation image is still a reasonable proxy for unseen data.
    train_files = sorted(p for b in train_bases for p in groups[b])
    val_files = sorted(sorted(groups[b])[0] for b in val_bases)
    test_files = sorted(sorted(groups[b])[0] for b in test_bases)
    print(f"[split] {len(train_files)} train images "
          f"({len(train_files) / max(len(train_bases), 1):.1f}x per source frame), "
          f"{len(val_files)} val, {len(test_files)} test")

    # Ultralytics resolves .txt list paths relative to the CWD at training
    # time, not relative to the YAML's `path:`. Writing absolute paths is the
    # least-surprising option — the dataset isn't portable across machines
    # without rerunning this script anyway.
    def write_list(name: str, files: list[Path]) -> Path:
        out = dataset / name
        out.write_text(
            "\n".join(str(p.resolve().as_posix()) for p in files) + "\n",
            encoding="utf-8",
        )
        return out

    train_txt = write_list("train_split.txt", train_files)
    val_txt = write_list("val_split.txt", val_files)
    test_txt = write_list("test_split.txt", test_files)

    # New data.yaml that points to the lists. Class names + count come from
    # the existing data.yaml so we don't have to re-derive them.
    original = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    new_yaml = {
        "path": str(dataset),
        "train": train_txt.name,
        "val": val_txt.name,
        "test": test_txt.name,
        "nc": original["nc"],
        "names": original["names"],
    }
    yaml_path = dataset / "data_split.yaml"
    yaml_path.write_text(yaml.safe_dump(new_yaml, sort_keys=False), encoding="utf-8")
    print(f"[done] {yaml_path}")
    print(f"        Use this with: yolo train data={yaml_path.relative_to(REPO_ROOT).as_posix()} ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
