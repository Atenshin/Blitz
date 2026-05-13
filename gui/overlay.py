"""Transparent overlay that paints detection bounding boxes on top of QVideoWidget.

This widget is a child of VideoPlayerWidget, sized to cover the entire player
area. QVideoWidget letterboxes video to preserve aspect ratio, so we compute
the actual video rect within the widget and map detection bboxes (which live
in source-video pixel coordinates) into widget coordinates.

For M3.5 this is view-only. M6 will add mouse interaction (drag corners, etc.)
without breaking the rendering path: paintEvent stays the same, we just add
mousePressEvent / mouseMoveEvent handlers and a "selected box" state.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from detection.cache_index import CacheIndex
from detection.schema import Detection


@dataclass(frozen=True)
class ClassStyle:
    color: QColor
    label_bg: QColor


# Class -> drawing style. Names must match what the trained model outputs.
# Match the spec: robots blue/red, ground balls red, airborne (counted shots)
# green. Goal gets a distinct purple to stand out from the action.
_STYLES: dict[str, ClassStyle] = {
    "robot_blue":    ClassStyle(QColor(60, 120, 255, 230), QColor(60, 120, 255, 200)),
    "robot_red":     ClassStyle(QColor(255, 60, 60, 230),  QColor(255, 60, 60, 200)),
    "ball_ground":   ClassStyle(QColor(255, 100, 50, 230), QColor(255, 100, 50, 200)),
    "ball_airborne": ClassStyle(QColor(80, 230, 80, 240),  QColor(60, 200, 60, 220)),
    "goal":          ClassStyle(QColor(220, 120, 255, 230), QColor(220, 120, 255, 200)),
}
_DEFAULT_STYLE = ClassStyle(QColor(200, 200, 200, 230), QColor(200, 200, 200, 200))

# Class -> visibility-toggle group used by keyboard shortcuts. R/G/B/O each
# toggle one group on/off.
_TOGGLE_GROUPS: dict[str, str] = {
    "robot_blue": "robots",
    "robot_red": "robots",
    "ball_ground": "ground_balls",
    "ball_airborne": "airborne_balls",
    "goal": "goal",
}


class DetectionOverlay(QWidget):
    """Paints boxes from a CacheIndex over its parent widget.

    Public API:
        set_cache(cache_index)            attach a new cache (or None to clear)
        set_video_size(w, h)              tell the overlay the source video's resolution
        set_current_sec(sec)              call this on every position update
        toggle_group(group_name)          flip visibility for a class group
        set_label_threshold(conf)         only draw labels for detections above this
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # Don't intercept mouse events — clicks should pass through to the
        # video widget. M6 will turn this off when entering edit mode.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Transparent background so the video shows through.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._cache: CacheIndex | None = None
        self._video_size: tuple[int, int] | None = None  # (w, h) in source pixels
        self._current_dets: list[Detection] = []
        self._visible_groups = {
            "robots": True,
            "ground_balls": True,
            "airborne_balls": True,
            "goal": True,
        }
        self._label_threshold = 0.0  # show all labels by default
        self._show_confidence = True

    # ----- public setters -----

    def set_cache(self, cache: CacheIndex | None) -> None:
        self._cache = cache
        self._current_dets = []
        self.update()

    def set_video_size(self, w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            self._video_size = None
        else:
            self._video_size = (w, h)
        self.update()

    def set_current_sec(self, sec: float) -> None:
        if self._cache is None:
            if self._current_dets:
                self._current_dets = []
                self.update()
            return
        frame = self._cache.find_frame_at(sec)
        new_dets = frame.detections if frame is not None else []
        # Only repaint when the detection list actually changes (Qt batches
        # paint events but checking saves QPainter setup cost).
        if new_dets is not self._current_dets:
            self._current_dets = new_dets
            self.update()

    def toggle_group(self, group: str) -> bool:
        """Returns the new visibility state."""
        if group not in self._visible_groups:
            return False
        self._visible_groups[group] = not self._visible_groups[group]
        self.update()
        return self._visible_groups[group]

    def is_group_visible(self, group: str) -> bool:
        return self._visible_groups.get(group, True)

    # ----- rendering -----

    def _video_rect_in_widget(self) -> tuple[float, float, float, float] | None:
        """Compute (offset_x, offset_y, scale_x, scale_y) for source -> widget
        coordinate mapping under QVideoWidget's KeepAspectRatio scaling."""
        if self._video_size is None:
            return None
        v_w, v_h = self._video_size
        w_w = self.width()
        w_h = self.height()
        if w_w <= 0 or w_h <= 0:
            return None
        scale = min(w_w / v_w, w_h / v_h)
        disp_w = v_w * scale
        disp_h = v_h * scale
        offset_x = (w_w - disp_w) / 2.0
        offset_y = (w_h - disp_h) / 2.0
        return (offset_x, offset_y, scale, scale)

    def paintEvent(self, _event) -> None:
        if not self._current_dets:
            return
        mapping = self._video_rect_in_widget()
        if mapping is None:
            return
        ox, oy, sx, sy = mapping

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        for det in self._current_dets:
            group = _TOGGLE_GROUPS.get(det.name, "")
            if group and not self._visible_groups.get(group, True):
                continue

            style = _STYLES.get(det.name, _DEFAULT_STYLE)
            x1, y1, x2, y2 = det.bbox
            rect = QRectF(
                x1 * sx + ox,
                y1 * sy + oy,
                (x2 - x1) * sx,
                (y2 - y1) * sy,
            )

            # Box outline.
            pen = QPen(style.color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            # Label background + text, if enabled.
            if det.conf < self._label_threshold:
                continue
            label = det.name
            if self._show_confidence:
                label = f"{det.name} {det.conf:.2f}"
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(label) + 6
            text_h = metrics.height() + 2
            label_rect = QRectF(rect.x(), rect.y() - text_h, text_w, text_h)
            painter.fillRect(label_rect, style.label_bg)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(
                label_rect.adjusted(3, 0, -3, 0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                label,
            )

        painter.end()
