"""Video player widget — QMediaPlayer + QVideoWidget (Path A).

Owns the QMediaPlayer instance. Other widgets (timeline scrubber, transport
controls, shortcut handler) talk to the player through this widget's public
methods rather than touching the QMediaPlayer directly.

A DetectionOverlay child widget is layered on top of QVideoWidget, sized to
match its rect. The overlay is transparent until a detection cache is attached
via `set_detection_cache()`.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QEvent, QSize, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaMetaData, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from detection.cache_index import CacheIndex
from .overlay import DetectionOverlay


class VideoPlayerWidget(QWidget):
    """Self-contained video surface.

    Public API:
        load(path)
        play() / pause() / toggle_play()
        seek_ms(ms) / seek_relative(delta_ms)
        step_frame(direction)            direction: +1 or -1
        set_rate(rate)                   0.25..2.0
        position(), duration(), is_playing()

    Signals:
        position_changed(int)      current playback position in ms
        duration_changed(int)      total duration in ms (once known)
        loaded(str)                emitted when a new file is loaded
    """

    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    loaded = pyqtSignal(str)

    # Assumed video FPS for frame stepping. Without decoding the file we don't
    # know the real value; 30 is right for nearly all FRC broadcasts and any
    # error here only affects single-frame nudges, not playback speed.
    _ASSUMED_FPS = 30

    def __init__(self):
        super().__init__()
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._video = QVideoWidget(self)
        self._player.setVideoOutput(self._video)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video)

        # Detection overlay sits on top of the video widget. We track
        # QVideoWidget's own resize events via an eventFilter so the overlay
        # follows it even when the layout resizes the child after our own
        # resizeEvent has fired.
        self.overlay = DetectionOverlay(self._video)
        self.overlay.raise_()
        self.overlay.show()
        self._video.installEventFilter(self)

        self._player.positionChanged.connect(self.position_changed.emit)
        self._player.durationChanged.connect(self.duration_changed.emit)
        # Route every position update into the overlay so it repaints with
        # detections for the nearest cached frame.
        self._player.positionChanged.connect(
            lambda ms: self.overlay.set_current_sec(ms / 1000.0)
        )
        # Pick up the video's native resolution when the file loads. The
        # overlay needs it to map source-pixel bboxes into widget coords.
        self._player.metaDataChanged.connect(self._on_metadata_changed)

    # --- transport ---

    def load(self, path: str | Path) -> None:
        p = Path(path)
        self._player.setSource(QUrl.fromLocalFile(str(p.resolve())))
        self.loaded.emit(str(p))

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
        """Approximate single-frame step. Pauses if currently playing."""
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
        # Force a repaint at the current position so overlays appear
        # immediately on pause/seek instead of waiting for the next tick.
        self.overlay.set_current_sec(self.position() / 1000.0)

    def _on_metadata_changed(self) -> None:
        # QMediaMetaData.Resolution is a QSize. It's populated some time after
        # the file loads, so this slot fires multiple times — only react when
        # we get something nonzero.
        size = self._player.metaData().value(QMediaMetaData.Key.Resolution)
        if isinstance(size, QSize) and size.isValid():
            self.overlay.set_video_size(size.width(), size.height())

    def eventFilter(self, watched, event):
        # When QVideoWidget resizes (whether from our own resizeEvent or from
        # the layout running after it), keep the overlay glued to its rect.
        if watched is self._video and event.type() == QEvent.Type.Resize:
            self.overlay.setGeometry(self._video.rect())
        return super().eventFilter(watched, event)
