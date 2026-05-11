"""GUI entry point."""
from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def run(repo_root: Path, cfg: dict, tba_auth_key: str | None) -> int:
    app = QApplication(sys.argv)
    win = MainWindow(repo_root, cfg, tba_auth_key or "")
    win.show()
    return app.exec()
