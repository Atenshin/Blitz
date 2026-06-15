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

from detection.cache_index import load_cache_for_video
from detection.identity import MatchIdentities

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
        self.detections_root = repo_root / cfg["paths"].get("detections", "detections")
        self.identities_root = repo_root / cfg["paths"].get("identities", "identities")
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

        self.bottom_panel = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_panel.addWidget(self._wrap("Matches", self.match_list))
        self.bottom_panel.addWidget(self._placeholder("Detection Controls", "Lands in Milestone 3"))
        self.bottom_panel.addWidget(self._placeholder("Object Editor", "Lands in Milestone 6"))
        self.bottom_panel.setSizes([320, 320, 320])

        self.root_split = QSplitter(Qt.Orientation.Vertical)
        self.root_split.addWidget(top)
        self.root_split.addWidget(self.bottom_panel)
        self.root_split.setSizes([560, 260])

        self.setCentralWidget(self.root_split)
        self._build_toolbar()
        self._wire_shortcuts()
        self._cinema_mode = False

        self.statusBar().showMessage("Ready")
        self.player.loaded.connect(lambda p: self.statusBar().showMessage(f"Loaded {Path(p).name}"))

    # --- chrome ---

    def _build_toolbar(self) -> None:
        self.toolbar = QToolBar("Main")
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        add_action = QAction("Add Videos…", self)
        add_action.triggered.connect(self._open_add_videos)
        self.toolbar.addAction(add_action)

        refresh_action = QAction("Refresh List", self)
        refresh_action.triggered.connect(self.match_list.refresh)
        self.toolbar.addAction(refresh_action)

        self.toolbar.addSeparator()

        cinema_action = QAction("Cinema Mode (Tab)", self)
        cinema_action.triggered.connect(self._toggle_cinema_mode)
        self.toolbar.addAction(cinema_action)

        fullscreen_action = QAction("Fullscreen (F11)", self)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        self.toolbar.addAction(fullscreen_action)

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
        bind("Tab", self._toggle_cinema_mode)
        bind("F11", self._toggle_fullscreen)
        bind("Escape", self._exit_fullscreen)
        # Detection-overlay visibility toggles. Spec called for R (robots) and
        # G (ground balls); we extend with B (airborne balls) and O (goal).
        bind("R", lambda: self._toggle_overlay_group("robots"))
        bind("G", lambda: self._toggle_overlay_group("ground_balls"))
        bind("B", lambda: self._toggle_overlay_group("airborne_balls"))
        bind("O", lambda: self._toggle_overlay_group("goal"))
        # Diagnostic: toggle a yellow "OVERLAY: N dets | W x H" badge in the
        # top-left so we can tell whether the overlay is rendering at all.
        bind("Shift+D", self._toggle_overlay_debug)

    def _toggle_overlay_group(self, group: str) -> None:
        now_visible = self.player.overlay.toggle_group(group)
        self.statusBar().showMessage(
            f"Overlay '{group}': {'visible' if now_visible else 'hidden'}", 1500
        )

    def _toggle_overlay_debug(self) -> None:
        overlay = self.player.overlay
        new = not overlay._debug
        overlay.set_debug(new)
        self.statusBar().showMessage(
            f"Overlay debug badge: {'on' if new else 'off'}", 2000
        )

    def _speed_up(self) -> None:
        c = self.timeline.speed_combo
        c.setCurrentIndex(min(c.count() - 1, c.currentIndex() + 1))

    def _speed_down(self) -> None:
        c = self.timeline.speed_combo
        c.setCurrentIndex(max(0, c.currentIndex() - 1))

    # --- handlers ---

    def _on_match_activated(self, path: str) -> None:
        video = Path(path)
        self.player.load(path)
        # Attach the detection cache if one exists for this video. The
        # overlay clears itself silently when None is passed.
        cache = load_cache_for_video(video, self.detections_root)
        self.player.set_detection_cache(cache)
        # Optional: load the identities file so team numbers appear on top
        # of tracked robots. Falls back to track-id labels if absent.
        identities = self._load_identities_for(video)
        self.player.overlay.set_identities(identities)

        status_parts = [f"Loaded {video.name}"]
        if cache is not None:
            n_dets = sum(len(f.detections) for f in cache.frames)
            status_parts.append(f"{len(cache.frames)} frames, {n_dets} detections")
        else:
            status_parts.append("no detection cache (run tools/run_inference.py)")
        if identities is not None:
            n_named = sum(1 for t in identities.tracks.values() if t.team_number)
            status_parts.append(f"{n_named} teams identified")
        self.statusBar().showMessage(" — ".join(status_parts), 5000)

        self.player.play()
        self.timeline.refresh_play_button()

    def _load_identities_for(self, video: Path) -> MatchIdentities | None:
        event_key = video.parent.name
        path = self.identities_root / event_key / f"{video.stem}.json"
        if not path.exists():
            return None
        try:
            return MatchIdentities.read(path)
        except Exception as e:
            self.statusBar().showMessage(f"Identities file failed to load: {e}", 5000)
            return None

    def _toggle_cinema_mode(self) -> None:
        """Hide toolbar + bottom panel for a giant video; Tab again to restore."""
        self._cinema_mode = not self._cinema_mode
        self.toolbar.setVisible(not self._cinema_mode)
        self.bottom_panel.setVisible(not self._cinema_mode)
        self.statusBar().setVisible(not self._cinema_mode)
        if not self._cinema_mode:
            self.statusBar().showMessage("Cinema mode off", 1500)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        # Hide everything except the video surface so it fills the whole screen.
        self.toolbar.setVisible(False)
        self.bottom_panel.setVisible(False)
        self.statusBar().setVisible(False)
        self.timeline.setVisible(False)
        self.showFullScreen()

    def _exit_fullscreen(self) -> None:
        if not self.isFullScreen():
            return
        self.showNormal()
        # Timeline is always visible in windowed mode.
        self.timeline.setVisible(True)
        # Toolbar, bottom panel, and status bar restore based on cinema mode.
        chrome_visible = not self._cinema_mode
        self.toolbar.setVisible(chrome_visible)
        self.bottom_panel.setVisible(chrome_visible)
        self.statusBar().setVisible(chrome_visible)

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
            allowed_uploaders=self.cfg["download"].get("allowed_uploaders") or [],
            default_event_key=self.cfg.get("event_key", ""),
        )
        dialog.downloads_completed.connect(self.match_list.refresh)
        dialog.show()
