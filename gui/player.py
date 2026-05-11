"""Video player widget — QMediaPlayer + QVideoWidget (Path A).

Owns the QMediaPlayer instance. Other widgets (timeline scrubber, transport
controls, shortcut handler) talk to the player through this widget's public
methods rather than touching the QMediaPlayer directly.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget


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

        self._player.positionChanged.connect(self.position_changed.emit)
        self._player.durationChanged.connect(self.duration_changed.emit)

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
