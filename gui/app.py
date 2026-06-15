"""GUI entry point."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the FFmpeg multimedia backend instead of the Windows-native WMF one.
# WMF renders video into a native HWND that sits above all Qt widgets in
# z-order, so transparent overlays (our detection bboxes) are hidden.
# FFmpeg renders into a regular Qt widget that composes correctly.
# This must be set BEFORE QApplication or any Qt multimedia import.
os.environ.setdefault("QT_MEDIA_BACKEND", "ffmpeg")

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def run(repo_root: Path, cfg: dict, tba_auth_key: str | None) -> int:
    app = QApplication(sys.argv)
    win = MainWindow(repo_root, cfg, tba_auth_key or "")
    win.show()
    return app.exec()
