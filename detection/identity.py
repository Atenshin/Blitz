"""Per-track team identity attribution.

Takes a detection cache (M3.4 + M4.1) and the source match metadata from
TBA, and produces a mapping `track_id -> {team_number, alliance, confidence}`
based on majority voting over OCR readings across the track's lifetime.

Output schema (one file per match at identities/<event>/<match_key>.json):

    {
      "schema_version": 1,
      "match_key": "2026cmptx_sf1m1",
      "red_teams": ["frc7407", "frc5940", "frc9470"],
      "blue_teams": ["frc868", "frc2910", "frc2046"],
      "samples_taken": 1234,
      "tracks": {
        "5": {
          "class_majority": "robot_red",
          "alliance": "red",
          "team_number": "9470",
          "confidence": 0.92,
          "n_readings": 18,
          "n_matched": 14
        },
        ...
      },
      "manual_overrides": {}    # populated by the GUI right-click reassign (M4.7)
    }

The "samples_taken" field is the total number of OCR calls made (one per
sampled robot bbox); useful for debugging coverage.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

IDENTITY_SCHEMA_VERSION = 1


@dataclass
class TrackIdentity:
    track_id: int
    class_majority: str          # "robot_blue" / "robot_red" / "ball_*" / "goal"
    alliance: Literal["red", "blue", "unknown"]
    team_number: str | None      # None if no team_number could be determined
    confidence: float            # share of validated readings that picked this team
    n_readings: int              # OCR readings collected for this track
    n_matched: int               # how many produced a valid roster-matching number


@dataclass
class MatchIdentities:
    schema_version: int
    match_key: str
    red_teams: list[str]
    blue_teams: list[str]
    samples_taken: int
    tracks: dict[int, TrackIdentity] = field(default_factory=dict)
    manual_overrides: dict[int, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "match_key": self.match_key,
                "red_teams": self.red_teams,
                "blue_teams": self.blue_teams,
                "samples_taken": self.samples_taken,
                "tracks": {
                    str(t.track_id): {
                        "class_majority": t.class_majority,
                        "alliance": t.alliance,
                        "team_number": t.team_number,
                        "confidence": round(t.confidence, 3),
                        "n_readings": t.n_readings,
                        "n_matched": t.n_matched,
                    }
                    for t in self.tracks.values()
                },
                "manual_overrides": {str(k): v for k, v in self.manual_overrides.items()},
            },
            indent=2,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def read(path: Path) -> "MatchIdentities":
        data = json.loads(path.read_text(encoding="utf-8"))
        tracks: dict[int, TrackIdentity] = {}
        for k, v in (data.get("tracks") or {}).items():
            tid = int(k)
            tracks[tid] = TrackIdentity(
                track_id=tid,
                class_majority=v["class_majority"],
                alliance=v["alliance"],
                team_number=v.get("team_number"),
                confidence=v.get("confidence", 0.0),
                n_readings=v.get("n_readings", 0),
                n_matched=v.get("n_matched", 0),
            )
        overrides = {
            int(k): v for k, v in (data.get("manual_overrides") or {}).items()
        }
        return MatchIdentities(
            schema_version=data.get("schema_version", 0),
            match_key=data["match_key"],
            red_teams=data.get("red_teams", []),
            blue_teams=data.get("blue_teams", []),
            samples_taken=data.get("samples_taken", 0),
            tracks=tracks,
            manual_overrides=overrides,
        )


def attribute_identity(
    track_id: int,
    class_counts: Counter[str],
    raw_readings: list[tuple[str, float]],
    red_teams_numeric: set[str],
    blue_teams_numeric: set[str],
) -> TrackIdentity:
    """Aggregate raw OCR readings for one track into a single TrackIdentity.

    Voting rule: each (text, conf) reading is validated against the alliance
    roster; only valid matches contribute votes. Confidence ties are broken
    by summed read-confidence so a single high-conf reading beats multiple
    low-conf ones.
    """
    class_majority = class_counts.most_common(1)[0][0] if class_counts else "unknown"
    # Which alliance roster do we validate against?
    valid: set[str]
    if class_majority == "robot_blue":
        alliance: Literal["red", "blue", "unknown"] = "blue"
        valid = blue_teams_numeric
    elif class_majority == "robot_red":
        alliance = "red"
        valid = red_teams_numeric
    else:
        # Not a robot — return early with no team number.
        return TrackIdentity(
            track_id=track_id,
            class_majority=class_majority,
            alliance="unknown",
            team_number=None,
            confidence=0.0,
            n_readings=len(raw_readings),
            n_matched=0,
        )

    # Tally roster-matching reads by team number, weighted by OCR confidence.
    weighted = defaultdict(float)
    n_matched = 0
    for text, conf in raw_readings:
        if text in valid:
            weighted[text] += conf
            n_matched += 1

    if not weighted:
        return TrackIdentity(
            track_id=track_id,
            class_majority=class_majority,
            alliance=alliance,
            team_number=None,
            confidence=0.0,
            n_readings=len(raw_readings),
            n_matched=0,
        )

    best_team, best_weight = max(weighted.items(), key=lambda kv: kv[1])
    total = sum(weighted.values())
    return TrackIdentity(
        track_id=track_id,
        class_majority=class_majority,
        alliance=alliance,
        team_number=best_team,
        confidence=best_weight / total,
        n_readings=len(raw_readings),
        n_matched=n_matched,
    )
