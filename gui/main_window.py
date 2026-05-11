"""Top-level QMainWindow.

Layout (M2 scope — bottom-center and bottom-right are placeholders that get
filled in M3/M6):

  Toolbar
  ┌────────────────────────────────────────────┐
  │           Video Player                      │
  ├────────────────────────────────────────────┤
  │   Timeline + transport + speed              │
  ├──────────┬───────────────┬─────────────────┤
  │  Match   │  Detection    │  Object         │
  │  List    │  Controls     │  Editor         │
  │ (live)   │  (M3 stub)    │  (M6 stub)      │
  └──────────┴───────────────┴─────────────────┘
"""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .add_videos_dialog import AddVideosDialog
from .match_list import MatchListWidget
from .player import VideoPlayerWidget
from .timeline import TimelineBar


class MainWindow(QMainWindow):
    def __init__(self, repo_root: Path, cfg: dict, tba_auth_key: str):
        super().__init__()
        self.setWindowTitle("FRC Match Analyzer")
        self.resize(1280, 820)

        self.repo_root = repo_root
        self.cfg = cfg
        self.tba_auth_key = tba_auth_key
        self.videos_root = repo_root / cfg["paths"]["videos"]
        self.videos_root.mkdir(parents=True, exist_ok=True)

        # --- widgets ---
        self.player = VideoPlayerWidget()
        self.timeline = TimelineBar(self.player)
        self.match_list = MatchListWidget(self.videos_root)
        self.match_list.match_activated.connect(self._on_match_activated)

        # Top: player. Below: timeline. Below: bottom panels split horizontally.
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.player, stretch=1)
        top_layout.addWidget(self.timeline)

        bottom = QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self._wrap("Matches", self.match_list))
        bottom.addWidget(self._placeholder("Detection Controls", "Lands in Milestone 3"))
        bottom.addWidget(self._placeholder("Object Editor", "Lands in Milestone 6"))
        bottom.setSizes([320, 320, 320])

        root_split = QSplitter(Qt.Orientation.Vertical)
        root_split.addWidget(top)
        root_split.addWidget(bottom)
        root_split.setSizes([560, 260])

        self.setCentralWidget(root_split)
        self._build_toolbar()
        self._wire_shortcuts()

        self.statusBar().showMessage("Ready")
        self.player.loaded.connect(lambda p: self.statusBar().showMessage(f"Loaded {Path(p).name}"))

    # --- chrome ---

    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        add_action = QAction("Add Videos…", self)
        add_action.triggered.connect(self._open_add_videos)
        bar.addAction(add_action)

        refresh_action = QAction("Refresh List", self)
        refresh_action.triggered.connect(self.match_list.refresh)
        bar.addAction(refresh_action)

    def _wrap(self, title: str, inner: QWidget) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(inner)
        return box

    def _placeholder(self, title: str, msg: str) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        label = QLabel(msg)
        label.setStyleSheet("color: gray; font-style: italic;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return box

    # --- shortcuts ---

    def _wire_shortcuts(self) -> None:
        def bind(seq: str, cb):
            sc = QShortcut(QKeySequence(seq), self)
            sc.activated.connect(cb)
            return sc

        bind("Space", lambda: (self.player.toggle_play(), self.timeline.refresh_play_button()))
        bind("A", lambda: self.player.step_frame(-1))
        bind("D", lambda: self.player.step_frame(+1))
        bind("Left", lambda: self.player.seek_relative(-5000))
        bind("Right", lambda: self.player.seek_relative(+5000))
        bind("Up", self._speed_up)
        bind("Down", self._speed_down)

    def _speed_up(self) -> None:
        c = self.timeline.speed_combo
        c.setCurrentIndex(min(c.count() - 1, c.currentIndex() + 1))

    def _speed_down(self) -> None:
        c = self.timeline.speed_combo
        c.setCurrentIndex(max(0, c.currentIndex() - 1))

    # --- handlers ---

    def _on_match_activated(self, path: str) -> None:
        self.player.load(path)
        self.player.play()
        self.timeline.refresh_play_button()

    def _open_add_videos(self) -> None:
        if not self.tba_auth_key:
            self.statusBar().showMessage(
                "TBA auth key missing — playlist/event downloads will fail. "
                "Single URL still works.", 8000
            )
        dialog = AddVideosDialog(
            self,
            videos_root=self.videos_root,
            format_spec=self.cfg["download"]["format"],
            retries=self.cfg["download"]["retries"],
            tba_auth_key=self.tba_auth_key or "",
            default_event_key=self.cfg.get("event_key", ""),
        )
        dialog.downloads_completed.connect(self.match_list.refresh)
        dialog.show()
