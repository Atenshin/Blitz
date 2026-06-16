"""YOLO inference for FRC matches.

Two entry points:
    FrameDetector(...).detect(frame) -> list[Detection]
        Used by the GUI for live single-frame inference / preview.
    process_video(video_path, ...) -> MatchDetectionCache
        Used by the CLI to bulk-process whole matches into the cache.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2

from .schema import Detection, FrameDetections, MatchDetectionCache, SCHEMA_VERSION


_ROBOT_CLASSES = {"robot_blue", "robot_red"}


def _passes_robot_filter(
    det: "Detection",
    min_width_px: float,
    min_height_px: float,
    min_aspect: float,
    max_aspect: float,
    min_confidence: float,
) -> bool:
    """Return False for robot detections that look like people, hats, or other
    non-robot objects.

    Robots have rectangular/trapezoidal bumpers and are large relative to the
    frame. People, hats, and coloured shirts are small and often tall-and-narrow.
    """
    if det.name not in _ROBOT_CLASSES:
        return True
    x1, y1, x2, y2 = det.bbox
    w, h = x2 - x1, y2 - y1
    if w < min_width_px or h < min_height_px:
        return False
    aspect = w / h if h > 0 else 0.0
    if aspect < min_aspect or aspect > max_aspect:
        return False
    if det.conf < min_confidence:
        return False
    return True


class FrameDetector:
    """Wraps an Ultralytics YOLO model with our cropped Detection schema.

    Loading the model is moderately slow (~1s); reuse a single instance
    across many frames or videos.

    Tracking: when `tracker` is set (default "bytetrack"), `detect()` calls
    `model.track()` with persistent state, so each Detection carries an
    `object_id` that's stable across frames of the same video. Call
    `reset_tracker()` between videos so IDs don't carry across matches.
    """

    def __init__(
        self,
        model_path: Path | str,
        imgsz: int = 1280,
        confidence: float = 0.35,
        iou: float = 0.5,
        device: int | str = 0,
        tracker: str = "bytetrack",  # "bytetrack" / "botsort" / "" to disable
        robot_filter: dict | None = None,
    ):
        # Late import so importing this module doesn't drag in torch/cv2
        # for callers who only need the schema.
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        # Treat the input as a "bare model name" (e.g. "yolov8m.pt") that
        # Ultralytics knows how to auto-download when the string has no
        # directory component. Local paths must exist on disk.
        is_bare_name = str(model_path) == self.model_path.name
        if not is_bare_name and not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                f"Train first with `python tools/train.py`."
            )

        self.model = YOLO(str(model_path))
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.tracker = tracker
        rf = robot_filter or {}
        self._rf_min_w = rf.get("min_width_px", 60)
        self._rf_min_h = rf.get("min_height_px", 40)
        self._rf_min_aspect = rf.get("min_aspect", 0.3)
        self._rf_max_aspect = rf.get("max_aspect", 5.0)
        self._rf_min_conf = rf.get("min_confidence", 0.45)
        # When tracking is on, the second call onward needs persist=True so
        # the tracker state carries across frames. The first call within a
        # video must use persist=False to reset prior video's state.
        self._track_persist = False

        # Ultralytics returns class names as a {idx: name} dict; convert to a
        # plain list ordered by index so the schema stays simple.
        names_dict = self.model.names
        self.class_names: list[str] = [
            names_dict[i] for i in sorted(names_dict.keys())
        ]

    def reset_tracker(self) -> None:
        """Call before the first frame of each new video so track IDs don't
        leak between matches."""
        self._track_persist = False

    def detect(self, frame) -> list[Detection]:
        """Run inference on a single BGR numpy frame. Returns detections in
        source-video pixel coordinates."""
        if self.tracker:
            results = self.model.track(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                tracker=f"{self.tracker}.yaml",
                persist=self._track_persist,
                verbose=False,
            )
            # After the first call, keep state for subsequent frames in the
            # same video. reset_tracker() flips this back to False.
            self._track_persist = True
        else:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )

        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        # r.boxes.id is None for detections the tracker couldn't match (e.g.
        # very brief appearances). Keep them in the cache but with object_id=None.
        ids = r.boxes.id.cpu().numpy().astype(int) if (
            self.tracker and r.boxes.id is not None
        ) else None

        out: list[Detection] = []
        for i, (bbox, conf, cls) in enumerate(zip(boxes, confs, clss)):
            obj_id = int(ids[i]) if ids is not None else None
            det = Detection(
                cls=int(cls),
                name=self.class_names[int(cls)],
                conf=float(conf),
                bbox=[float(x) for x in bbox.tolist()],
                object_id=obj_id,
            )
            if not _passes_robot_filter(
                det,
                self._rf_min_w, self._rf_min_h,
                self._rf_min_aspect, self._rf_max_aspect,
                self._rf_min_conf,
            ):
                continue
            out.append(det)
        return out


def _iterate_sampled_frames(cap, frame_step: int) -> Iterator[tuple[int, "cv2.Mat"]]:
    """Yield (frame_idx, frame) for every Nth source frame.

    Uses grab() to skip past frames we don't need (cheap) and retrieve()
    only on the ones we want (expensive). Matches the optimization in
    tools/extract_frames.py.
    """
    frame_idx = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            return
        if frame_idx % frame_step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield frame_idx, frame
        frame_idx += 1


def process_video(
    video_path: Path,
    detector: FrameDetector,
    inference_fps: float = 15.0,
    repo_root: Path | None = None,
    progress_cb=None,
) -> MatchDetectionCache:
    """Run the detector over a whole video, returning the populated cache.

    progress_cb, if given, is called as `progress_cb(sec_processed, total_sec)`
    every N detections, suitable for driving a progress bar.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps if src_fps else 0.0
    frame_step = max(1, int(round(src_fps / inference_fps)))

    match_key = video_path.stem
    try:
        video_rel = str(video_path.resolve().relative_to(repo_root).as_posix())
    except ValueError:
        video_rel = str(video_path)
    try:
        model_rel = str(detector.model_path.resolve().relative_to(repo_root).as_posix())
    except ValueError:
        model_rel = str(detector.model_path)

    cache = MatchDetectionCache(
        schema_version=SCHEMA_VERSION,
        match_key=match_key,
        video=video_rel,
        model=model_rel,
        model_classes=list(detector.class_names),
        src_fps=src_fps,
        src_total_frames=total_frames,
        inference_fps=inference_fps,
        frame_step=frame_step,
        imgsz=detector.imgsz,
        confidence=detector.confidence,
        iou=detector.iou,
        tracking_used=bool(detector.tracker),
        tracker=detector.tracker,
    )

    # Reset ByteTrack state so IDs from the previous video don't carry over.
    detector.reset_tracker()

    last_progress = 0.0
    for frame_idx, frame in _iterate_sampled_frames(cap, frame_step):
        detections = detector.detect(frame)
        sec = frame_idx / src_fps
        cache.frames.append(FrameDetections(
            frame_idx=frame_idx,
            sec=sec,
            detections=detections,
        ))
        # Throttle the callback so the bar updates ~10x/sec instead of every frame.
        if progress_cb is not None and (sec - last_progress) >= 0.1:
            progress_cb(sec, duration)
            last_progress = sec

    cap.release()
    if progress_cb is not None:
        progress_cb(duration, duration)
    return cache


def cache_path_for(video_path: Path, detections_root: Path) -> Path:
    """Where the cache file lives for a given match video."""
    event = video_path.parent.name
    return detections_root / event / f"{video_path.stem}.json"
