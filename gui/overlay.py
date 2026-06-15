"""Detection overlay — a single QGraphicsItem that paints every bounding box
for the currently-visible frame inside the video player's QGraphicsScene.

This used to be a transparent child QWidget on top of QVideoWidget. That
pattern doesn't work on Windows: QVideoWidget renders into a native window
that ignores Qt's z-order, so the overlay was always hidden. Putting both
video (QGraphicsVideoItem) and overlay (this class) into a shared scene
solves the compositing problem cleanly — Qt handles z-order and the
viewport's transform scales both items together, so we draw boxes in raw
source-pixel coordinates without any manual scaling math.
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QGraphicsItem

from detection.cache_index import CacheIndex
from detection.identity import MatchIdentities, TrackIdentity
from detection.schema import Detection


@dataclass(frozen=True)
class ClassStyle:
    color: QColor
    label_bg: QColor


# Match the spec: robots in alliance colors, ground balls red, airborne
# (counted shots) green. Goal gets purple to stand out from the action.
_STYLES: dict[str, ClassStyle] = {
    "robot_blue":    ClassStyle(QColor(60, 120, 255, 230), QColor(60, 120, 255, 200)),
    "robot_red":     ClassStyle(QColor(255, 60, 60, 230),  QColor(255, 60, 60, 200)),
    "ball_ground":   ClassStyle(QColor(255, 100, 50, 230), QColor(255, 100, 50, 200)),
    "ball_airborne": ClassStyle(QColor(80, 230, 80, 240),  QColor(60, 200, 60, 220)),
    "goal":          ClassStyle(QColor(220, 120, 255, 230), QColor(220, 120, 255, 200)),
}
_DEFAULT_STYLE = ClassStyle(QColor(200, 200, 200, 230), QColor(200, 200, 200, 200))

# Class -> visibility-toggle group used by keyboard shortcuts.
_TOGGLE_GROUPS: dict[str, str] = {
    "robot_blue": "robots",
    "robot_red": "robots",
    "ball_ground": "ground_balls",
    "ball_airborne": "airborne_balls",
    "goal": "goal",
}


class DetectionOverlay(QGraphicsItem):
    """Lives in the player's QGraphicsScene at z=10, above the video item.

    Public API (kept stable so main_window.py doesn't change):
        set_cache(cache_index)
        set_video_size(w, h)
        set_current_sec(sec)
        toggle_group(group_name)
        set_debug(on)
    """

    def __init__(self):
        super().__init__()
        self.setZValue(10)  # above the video item (which sits at default z=0)
        # We don't accept mouse events during M3.5 (view-only). M6 will flip
        # this on to enable click-to-select for editing.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._cache: CacheIndex | None = None
        self._identities: MatchIdentities | None = None
        self._video_size: tuple[int, int] | None = None
        self._current_dets: list[Detection] = []
        self._visible_groups = {
            "robots": True,
            "ground_balls": True,
            "airborne_balls": True,
            "goal": True,
        }
        self._label_threshold = 0.0
        self._show_confidence = True
        # Toggle on with set_debug(True) (or Shift+D in the GUI) for the
        # magenta status badge — useful when rendering looks broken.
        self._debug = False

    # ----- public setters -----

    def set_cache(self, cache: CacheIndex | None) -> None:
        self._cache = cache
        self._current_dets = []
        self.update()

    def set_identities(self, identities: MatchIdentities | None) -> None:
        """Attach a MatchIdentities file so robot labels render as team
        numbers (e.g. "4499") instead of track IDs ("robot_blue #5")."""
        self._identities = identities
        self.update()

    def _team_label_for(self, det: Detection) -> str | None:
        """Look up the team number for a detection. Returns None if the
        detection has no track ID, or no identities are loaded, or no team
        could be attributed to that track."""
        if self._identities is None or det.object_id is None:
            return None
        # Manual overrides win over OCR-derived attribution.
        override = self._identities.manual_overrides.get(det.object_id)
        if override:
            return override
        ident = self._identities.tracks.get(det.object_id)
        if ident is None:
            return None
        return ident.team_number  # may be None if OCR found nothing valid

    def set_video_size(self, w: int, h: int) -> None:
        self.prepareGeometryChange()  # required when boundingRect changes
        self._video_size = (w, h) if (w > 0 and h > 0) else None
        self.update()

    def set_current_sec(self, sec: float) -> None:
        if self._cache is None:
            if self._current_dets:
                self._current_dets = []
                self.update()
            return
        frame = self._cache.find_frame_at(sec)
        new_dets = frame.detections if frame is not None else []
        if new_dets is not self._current_dets:
            self._current_dets = new_dets
            self.update()

    def toggle_group(self, group: str) -> bool:
        if group not in self._visible_groups:
            return False
        self._visible_groups[group] = not self._visible_groups[group]
        self.update()
        return self._visible_groups[group]

    def is_group_visible(self, group: str) -> bool:
        return self._visible_groups.get(group, True)

    def set_debug(self, on: bool) -> None:
        self._debug = on
        self.update()

    # ----- QGraphicsItem implementation -----

    def boundingRect(self) -> QRectF:
        """Item's drawing area in scene coordinates.

        Since we draw boxes in source-pixel coords and the scene is set up
        to match source resolution, our rect is just the full video frame.
        """
        if self._video_size is None:
            return QRectF(0, 0, 1, 1)
        w, h = self._video_size
        return QRectF(0, 0, w, h)

    def paint(self, painter: QPainter, _option, _widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._debug and self._video_size is not None:
            # Yellow status badge at top-left of the video frame, in scene
            # coords. Toggle with Shift+D when rendering looks off.
            w, h = self._video_size
            badge_w = w * 0.20
            badge_h = h * 0.05
            badge = QRectF(w * 0.01, h * 0.01, badge_w, badge_h)
            painter.fillRect(badge, QColor(255, 230, 0, 220))
            font = QFont(painter.font())
            font.setBold(True)
            font.setPointSizeF(badge_h * 0.45)
            painter.setFont(font)
            painter.setPen(QPen(QColor(0, 0, 0)))
            ndets = len(self._current_dets)
            painter.drawText(
                badge.adjusted(badge_h * 0.2, 0, -badge_h * 0.2, 0),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                f"{ndets} dets  |  {w}x{h}",
            )

        if not self._current_dets:
            return

        # Detection boxes — drawn directly in source-video pixel coordinates.
        # No scaling math needed; QGraphicsView's transform handles it.
        font = QFont(painter.font())
        font.setBold(True)
        font.setPointSizeF(max(8.0, (self._video_size[1] if self._video_size else 1080) * 0.012))
        painter.setFont(font)

        for det in self._current_dets:
            group = _TOGGLE_GROUPS.get(det.name, "")
            if group and not self._visible_groups.get(group, True):
                continue

            style = _STYLES.get(det.name, _DEFAULT_STYLE)
            x1, y1, x2, y2 = det.bbox
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            pen = QPen(style.color)
            pen.setWidth(3)
            pen.setCosmetic(True)  # constant pixel width regardless of zoom
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)

            if det.conf < self._label_threshold:
                continue
            # Label priority (most informative first):
            #   1. Attributed team number ("4499") — both robot color comes
            #      from the box, and the number is the persistent identity
            #   2. Track ID ("robot_blue #5") if we have a track but no team
            #   3. Class + confidence for un-tracked detections
            team = self._team_label_for(det)
            if team is not None:
                label = team
            elif det.object_id is not None:
                label = f"{det.name} #{det.object_id}"
            elif self._show_confidence:
                label = f"{det.name} {det.conf:.2f}"
            else:
                label = det.name
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
