"""Timeline scrubber and transport-controls strip."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .player import VideoPlayerWidget


def _ms_to_clock(ms: int) -> str:
    if ms <= 0:
        return "0:00"
    s = ms // 1000
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class TimelineBar(QWidget):
    """Slider + transport buttons + speed selector, bound to a VideoPlayerWidget."""

    _SPEEDS = [0.25, 0.5, 1.0, 1.5, 2.0]

    def __init__(self, player: VideoPlayerWidget):
        super().__init__()
        self.player = player

        # --- scrubber row ---
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setSingleStep(1000)        # 1s
        self.slider.setPageStep(10000)         # 10s
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.sliderReleased.connect(self._on_slider_released)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(110)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        scrub_row = QHBoxLayout()
        scrub_row.addWidget(self.slider, stretch=1)
        scrub_row.addWidget(self.time_label)

        # --- transport row ---
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(44)
        self.play_btn.clicked.connect(self.player.toggle_play)

        self.back_5_btn = QPushButton("−5s")
        self.fwd_5_btn = QPushButton("+5s")
        self.back_5_btn.clicked.connect(lambda: self.player.seek_relative(-5000))
        self.fwd_5_btn.clicked.connect(lambda: self.player.seek_relative(+5000))

        self.prev_frame_btn = QPushButton("◀ Frame")
        self.next_frame_btn = QPushButton("Frame ▶")
        self.prev_frame_btn.clicked.connect(lambda: self.player.step_frame(-1))
        self.next_frame_btn.clicked.connect(lambda: self.player.step_frame(+1))

        self.speed_combo = QComboBox()
        for s in self._SPEEDS:
            self.speed_combo.addItem(f"{s}×", s)
        self.speed_combo.setCurrentIndex(self._SPEEDS.index(1.0))
        self.speed_combo.currentIndexChanged.connect(
            lambda i: self.player.set_rate(self.speed_combo.itemData(i))
        )

        # --- volume ---
        # Mute button toggles audio; its glyph reflects mute/level state.
        self.mute_btn = QPushButton("🔊")
        self.mute_btn.setFixedWidth(36)
        self.mute_btn.setCheckable(True)
        self.mute_btn.clicked.connect(self._on_mute_toggled)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(int(self.player.volume() * 100))
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)

        controls = QHBoxLayout()
        for w in (self.back_5_btn, self.prev_frame_btn, self.play_btn,
                  self.next_frame_btn, self.fwd_5_btn):
            controls.addWidget(w)
        controls.addStretch(1)
        controls.addWidget(self.mute_btn)
        controls.addWidget(self.volume_slider)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Speed:"))
        controls.addWidget(self.speed_combo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.addLayout(scrub_row)
        layout.addLayout(controls)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # --- wire player → UI ---
        self.player.position_changed.connect(self._on_position_changed)
        self.player.duration_changed.connect(self._on_duration_changed)

    # --- player → UI ---

    def _on_position_changed(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self._update_label(ms, self.slider.maximum())

    def _on_duration_changed(self, ms: int) -> None:
        self.slider.setRange(0, ms)
        self._update_label(self.player.position(), ms)

    def _update_label(self, pos: int, dur: int) -> None:
        self.time_label.setText(f"{_ms_to_clock(pos)} / {_ms_to_clock(dur)}")

    # --- UI → player ---

    def _on_slider_moved(self, value: int) -> None:
        # Update the label live during drag, but don't seek the player every tick
        # (that thrashes the decoder). The actual seek happens on release.
        self._update_label(value, self.slider.maximum())

    def _on_slider_released(self) -> None:
        self.player.seek_ms(self.slider.value())

    # --- volume ---

    def _on_volume_changed(self, value: int) -> None:
        self.player.set_volume(value / 100.0)
        # Dragging the slider above zero implicitly unmutes.
        if value > 0 and self.mute_btn.isChecked():
            self.mute_btn.setChecked(False)
            self.player.set_muted(False)
        self._refresh_mute_glyph()

    def _on_mute_toggled(self, muted: bool) -> None:
        self.player.set_muted(muted)
        self._refresh_mute_glyph()

    def _refresh_mute_glyph(self) -> None:
        if self.mute_btn.isChecked() or self.volume_slider.value() == 0:
            self.mute_btn.setText("🔇")
        elif self.volume_slider.value() < 50:
            self.mute_btn.setText("🔉")
        else:
            self.mute_btn.setText("🔊")

    # Called by the main window to reflect external play/pause changes
    # (e.g. the Space shortcut) on the button label.
    def refresh_play_button(self) -> None:
        self.play_btn.setText("⏸" if self.player.is_playing() else "▶")
