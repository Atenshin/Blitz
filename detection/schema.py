"""Cache-file schema for per-match detections.

Single source of truth shared by inference (M3.4), GUI overlay rendering
(M3.5), tracking (M4), and the manual editor (M6). When the schema evolves,
bump SCHEMA_VERSION so older caches can be detected and either migrated or
regenerated.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

SCHEMA_VERSION = 1


@dataclass
class Detection:
    cls: int                       # class index from the model
    name: str                      # class name (e.g. "robot_blue")
    conf: float                    # confidence 0-1
    bbox: list[float]              # [x1, y1, x2, y2] in source-video pixel coords
    source: Literal["yolo", "manual"] = "yolo"
    object_id: int | None = None   # populated by the tracker (M4); null here


@dataclass
class FrameDetections:
    frame_idx: int                 # 0-based index in the source video
    sec: float                     # frame_idx / src_fps
    detections: list[Detection] = field(default_factory=list)


@dataclass
class MatchDetectionCache:
    schema_version: int
    match_key: str
    video: str                     # path relative to repo root
    model: str                     # path relative to repo root
    model_classes: list[str]       # in class-index order
    src_fps: float                 # video's native frame rate
    src_total_frames: int          # total frames in the video file
    inference_fps: float           # requested cadence (frames sampled per second)
    frame_step: int                # we run detection every N source frames
    imgsz: int
    confidence: float
    iou: float
    frames: list[FrameDetections] = field(default_factory=list)

    # ---- IO helpers ----

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def read(path: Path) -> "MatchDetectionCache":
        data = json.loads(path.read_text(encoding="utf-8"))
        frames = [
            FrameDetections(
                frame_idx=f["frame_idx"],
                sec=f["sec"],
                detections=[Detection(**d) for d in f["detections"]],
            )
            for f in data.get("frames", [])
        ]
        return MatchDetectionCache(
            schema_version=data.get("schema_version", 0),
            match_key=data["match_key"],
            video=data["video"],
            model=data["model"],
            model_classes=data["model_classes"],
            src_fps=data["src_fps"],
            src_total_frames=data["src_total_frames"],
            inference_fps=data["inference_fps"],
            frame_step=data["frame_step"],
            imgsz=data["imgsz"],
            confidence=data["confidence"],
            iou=data["iou"],
            frames=frames,
        )
