"""Sidebar tree of locally cached matches, grouped by event."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHeaderView, QTreeWidget, QTreeWidgetItem


class MatchListWidget(QTreeWidget):
    """Tree of events → matches.

    Each match row stores its absolute video path on the item's UserRole.
    Items for matches that haven't been downloaded yet are still shown but
    grayed out, so the user can see what's available.
    """

    match_activated = pyqtSignal(str)   # absolute path to .mp4 (or "" if missing)

    def __init__(self, videos_root: Path):
        super().__init__()
        self.videos_root = videos_root
        self.setHeaderLabels(["Match", "Status"])
        self.setColumnWidth(0, 240)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.itemActivated.connect(self._on_activated)
        self.itemDoubleClicked.connect(self._on_activated)
        self.refresh()

    def refresh(self) -> None:
        """Re-scan videos/ from disk. Safe to call whenever a download finishes."""
        self.clear()
        if not self.videos_root.exists():
            return
        for event_dir in sorted(p for p in self.videos_root.iterdir() if p.is_dir()):
            if event_dir.name.startswith("_"):
                # _unassigned and friends — show under a special group at the bottom.
                continue
            self._add_event(event_dir)

        unassigned = self.videos_root / "_unassigned"
        if unassigned.exists() and any(unassigned.iterdir()):
            self._add_unassigned(unassigned)

        self.expandAll()

    def _add_event(self, event_dir: Path) -> None:
        matches_json = event_dir / "matches.json"
        event_item = QTreeWidgetItem([event_dir.name, ""])
        event_item.setFlags(event_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.addTopLevelItem(event_item)

        if matches_json.exists():
            try:
                data = json.loads(matches_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = []
            for m in data:
                key = m["key"]
                short = m.get("short_name", key)
                video_path = event_dir / f"{key}.mp4"
                exists = video_path.exists()
                child = QTreeWidgetItem([short, "✓" if exists else "⬇"])
                child.setData(0, Qt.ItemDataRole.UserRole, str(video_path) if exists else "")
                if not exists:
                    child.setForeground(0, Qt.GlobalColor.gray)
                    child.setForeground(1, Qt.GlobalColor.gray)
                event_item.addChild(child)
        else:
            # No TBA metadata cached — just list mp4s on disk.
            for mp4 in sorted(event_dir.glob("*.mp4")):
                child = QTreeWidgetItem([mp4.stem, "✓"])
                child.setData(0, Qt.ItemDataRole.UserRole, str(mp4))
                event_item.addChild(child)

    def _add_unassigned(self, unassigned_dir: Path) -> None:
        event_item = QTreeWidgetItem(["(unassigned)", ""])
        event_item.setFlags(event_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.addTopLevelItem(event_item)
        for mp4 in sorted(unassigned_dir.glob("*.mp4")):
            child = QTreeWidgetItem([mp4.stem, "✓"])
            child.setData(0, Qt.ItemDataRole.UserRole, str(mp4))
            event_item.addChild(child)

    def _on_activated(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.match_activated.emit(path)
