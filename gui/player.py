"""Video player — QGraphicsView + QGraphicsVideoItem.

We started with QVideoWidget + a transparent child overlay (Path A), but on
Windows + Qt6, QVideoWidget's native render surface covers any Qt widget on
top regardless of z-order. The graphics-scene approach puts both the video
(QGraphicsVideoItem) and the detection overlay (DetectionOverlay, a custom
QGraphicsItem) into the same scene; the view's transform scales everything
together, and Qt composes z-order naturally because it's all inside one
QPainter pipeline.

Public API is unchanged so callers in main_window.py and timeline.py don't
need to change.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRectF, QSizeF, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from detection.cache_index import CacheIndex
from .overlay import DetectionOverlay


class VideoPlayerWidget(QWidget):
    """Self-contained video surface.

    Public API:
        load(path)
        play() / pause() / toggle_play()
        seek_ms(ms) / seek_relative(delta_ms)
        step_frame(direction)
        set_rate(rate)
        position(), duration(), is_playing()
        set_detection_cache(cache_index)
        overlay                         # the DetectionOverlay graphics item

    Signals:
        position_changed(int)
        duration_changed(int)
        loaded(str)
    """

    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    loaded = pyqtSignal(str)

    _ASSUMED_FPS = 30

    def __init__(self):
        super().__init__()

        # --- scene + items ---
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QBrush(QColor(0, 0, 0)))

        # Video item lives at z=0. We'll size it to the source video's
        # resolution once we know it (via _probe_video_size).
        self._video_item = QGraphicsVideoItem()
        self._scene.addItem(self._video_item)

        # Detection overlay sits at z=10 — same coordinate system as the
        # video. No manual scaling math; we draw boxes in raw bbox pixels.
        self.overlay = DetectionOverlay()
        self._scene.addItem(self.overlay)

        # --- view ---
        self._view = QGraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._view.setFrameStyle(0)
        self._view.setStyleSheet("background: black; border: none;")
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        # Pin the scene to top-left so the video doesn't drift when sized.
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- player ---
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_item)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._player.positionChanged.connect(self.position_changed.emit)
        self._player.durationChanged.connect(self.duration_changed.emit)
        self._player.positionChanged.connect(
            lambda ms: self.overlay.set_current_sec(ms / 1000.0)
        )

        # Until we know the real resolution, give the scene a sensible
        # default so the view has something to fit.
        self._set_scene_size(1920, 1080)

    # --- transport ---

    def load(self, path: str | Path) -> None:
        p = Path(path)
        # Read source resolution via cv2 — fast and reliable, unlike
        # QMediaPlayer.metaDataChanged which is unreliable on Windows.
        size = self._probe_video_size(p)
        if size is not None:
            self._set_scene_size(*size)
        self._player.setSource(QUrl.fromLocalFile(str(p.resolve())))
        self.loaded.emit(str(p))

    def _probe_video_size(self, video_path: Path) -> tuple[int, int] | None:
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return (w, h)
        except Exception:
            pass
        return None

    def _set_scene_size(self, w: int, h: int) -> None:
        """Reshape the scene + video item to match source resolution and
        re-fit the view so everything stays centered and aspect-correct."""
        self._video_item.setSize(QSizeF(w, h))
        self._scene.setSceneRect(0, 0, w, h)
        self.overlay.set_video_size(w, h)
        self._fit_view()

    def _fit_view(self) -> None:
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def play(self) -> None:
        self._player.play()

    def pause(self) -> None:
        self._player.pause()

    def toggle_play(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def seek_ms(self, ms: int) -> None:
        ms = max(0, min(ms, self.duration() or ms))
        self._player.setPosition(ms)

    def seek_relative(self, delta_ms: int) -> None:
        self.seek_ms(self.position() + delta_ms)

    def step_frame(self, direction: int) -> None:
        if self.is_playing():
            self.pause()
        delta = int(1000 / self._ASSUMED_FPS) * (1 if direction > 0 else -1)
        self.seek_relative(delta)

    def set_rate(self, rate: float) -> None:
        self._player.setPlaybackRate(max(0.05, rate))

    def rate(self) -> float:
        return self._player.playbackRate()

    def position(self) -> int:
        return self._player.position()

    def duration(self) -> int:
        return self._player.duration()

    # --- detection overlay integration ---

    def set_detection_cache(self, cache: CacheIndex | None) -> None:
        self.overlay.set_cache(cache)
        # Trigger an immediate refresh at the current position so overlays
        # appear without waiting for the next position tick.
        self.overlay.set_current_sec(self.position() / 1000.0)

    # --- view fitting ---

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_view()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit_view()
