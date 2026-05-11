"""Add Videos dialog — three tabs covering single URL / playlist / event."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .download_worker import DownloadWorker


class AddVideosDialog(QDialog):
    """Modal-ish dialog. Doesn't actually block — the user can keep watching
    a video in the main window while a download runs.

    Emits `downloads_completed()` whenever a batch finishes (success or
    partial), so the main window can refresh the match list.
    """

    downloads_completed = pyqtSignal()

    def __init__(
        self,
        parent,
        videos_root: Path,
        format_spec: str,
        retries: int,
        tba_auth_key: str,
        default_event_key: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Add Videos")
        self.setMinimumSize(640, 460)

        self.videos_root = videos_root
        self.format_spec = format_spec
        self.retries = retries
        self.tba_auth_key = tba_auth_key

        # --- tabs ---
        self.tabs = QTabWidget()
        self._build_url_tab()
        self._build_playlist_tab(default_event_key)
        self._build_event_tab(default_event_key)

        # --- progress + log ---
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)

        self.start_btn = QPushButton("Start Download")
        self.start_btn.clicked.connect(self._on_start)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)

        button_row = QHBoxLayout()
        button_row.addWidget(self.start_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log, stretch=1)
        layout.addLayout(button_row)

        self._thread: QThread | None = None
        self._worker: DownloadWorker | None = None

    # --- tab builders ---

    def _build_url_tab(self) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://www.youtube.com/watch?v=…")
        self.url_match_key = QLineEdit()
        self.url_match_key.setPlaceholderText("optional, e.g. 2025miket_qm12")
        form.addRow("YouTube URL:", self.url_input)
        form.addRow("Save as match key:", self.url_match_key)
        form.addRow(QLabel(
            "<i>Leave the match key empty to save under videos/_unassigned/.</i>"
        ))
        self.tabs.addTab(w, "Single Video")

    def _build_playlist_tab(self, default_event_key: str) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.playlist_input = QLineEdit()
        self.playlist_input.setPlaceholderText("https://www.youtube.com/playlist?list=…")
        self.playlist_event = QLineEdit(default_event_key)
        self.playlist_event.setPlaceholderText("e.g. 2025miket")
        form.addRow("Playlist URL:", self.playlist_input)
        form.addRow("TBA event key:", self.playlist_event)
        form.addRow(QLabel(
            "<i>Titles like 'Qualification 12' are auto-mapped to the event's "
            "TBA matches. Unmatched videos are skipped (logged below).</i>"
        ))
        self.tabs.addTab(w, "Playlist + Event")

    def _build_event_tab(self, default_event_key: str) -> None:
        w = QWidget()
        form = QFormLayout(w)
        self.event_input = QLineEdit(default_event_key)
        self.event_input.setPlaceholderText("e.g. 2025miket")
        form.addRow("TBA event key:", self.event_input)
        form.addRow(QLabel(
            "<i>Uses YouTube links TBA has already stored for the event. "
            "Best option when available — no playlist needed.</i>"
        ))
        self.tabs.addTab(w, "Event Code Only")

    # --- start ---

    def _on_start(self) -> None:
        if self._thread is not None:
            return  # download already in progress
        idx = self.tabs.currentIndex()

        # Validate inputs based on selected tab.
        if idx == 0:
            url = self.url_input.text().strip()
            if not url:
                self._log("Enter a YouTube URL first.")
                return
            mode = ("single", url, self.url_match_key.text().strip() or None)
        elif idx == 1:
            url = self.playlist_input.text().strip()
            event = self.playlist_event.text().strip()
            if not url or not event:
                self._log("Both playlist URL and event key are required.")
                return
            mode = ("playlist", url, event)
        else:
            event = self.event_input.text().strip()
            if not event:
                self._log("Enter a TBA event key.")
                return
            mode = ("event", event)

        # Spin up the worker thread.
        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self._spawn_worker(mode)

    def _spawn_worker(self, mode: tuple) -> None:
        self._thread = QThread(self)
        self._worker = DownloadWorker(
            videos_root=self.videos_root,
            format_spec=self.format_spec,
            retries=self.retries,
            tba_auth_key=self.tba_auth_key,
        )
        self._worker.moveToThread(self._thread)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._log)
        self._worker.finished.connect(self._on_finished)

        # Map mode → method on the worker, kicked off when the thread starts.
        if mode[0] == "single":
            _, url, key = mode
            self._thread.started.connect(lambda: self._worker.run_single(url, key))
        elif mode[0] == "playlist":
            _, url, event = mode
            self._thread.started.connect(lambda: self._worker.run_playlist(url, event))
        else:
            _, event = mode
            self._thread.started.connect(lambda: self._worker.run_event(event))

        self._thread.start()

    # --- signal handlers ---

    def _on_progress(self, msg: str, pct: float) -> None:
        if pct >= 0:
            self.progress.setValue(int(pct))
            self.progress.setFormat(f"{msg} — %p%")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(msg)

    def _log(self, line: str) -> None:
        self.log.appendPlainText(line)

    def _on_finished(self, succeeded: int, failed: int, errors: list[str]) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat(f"Done — {succeeded} ok, {failed} failed")
        self._log(f"\n=== finished: {succeeded} succeeded, {failed} failed ===")
        for e in errors:
            self._log(e)
        self.downloads_completed.emit()

        # Tear down the thread.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self.start_btn.setEnabled(True)
