"""The Blue Alliance API v3 client."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import yaml

TBA_BASE = "https://www.thebluealliance.com/api/v3"


@dataclass
class MatchVideo:
    type: str   # "youtube" | "tba"
    key: str    # youtube video id when type == "youtube"

    @property
    def youtube_url(self) -> str | None:
        if self.type == "youtube":
            return f"https://www.youtube.com/watch?v={self.key}"
        return None


@dataclass
class Match:
    key: str                       # e.g. "2025miket_qm12"
    event_key: str                 # e.g. "2025miket"
    comp_level: str                # qm | ef | qf | sf | f | pm
    set_number: int
    match_number: int
    red_teams: list[str] = field(default_factory=list)   # ["frc1234", ...]
    blue_teams: list[str] = field(default_factory=list)
    videos: list[MatchVideo] = field(default_factory=list)

    @property
    def short_name(self) -> str:
        """Human-friendly: 'Qualification 12', 'Playoff 5 Match 1', 'Final 1'."""
        level_names = {
            "qm": "Qualification",
            "ef": "Eighth-Final",
            "qf": "Quarterfinal",
            "sf": "Playoff",
            "f": "Final",
            "pm": "Practice",
        }
        name = level_names.get(self.comp_level, self.comp_level)
        if self.comp_level in ("qm", "pm"):
            return f"{name} {self.match_number}"
        return f"{name} {self.set_number} Match {self.match_number}"


class TBAClient:
    def __init__(self, auth_key: str, cache_dir: Path | None = None):
        self.auth_key = auth_key
        self.session = requests.Session()
        self.session.headers.update({"X-TBA-Auth-Key": auth_key})
        self.cache_dir = cache_dir

    def _get(self, path: str) -> Any:
        url = f"{TBA_BASE}{path}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def event_matches(self, event_key: str) -> list[Match]:
        """Fetch all matches for an event with full data (including video links)."""
        data = self._get(f"/event/{event_key}/matches")
        matches = [_parse_match(m) for m in data]
        matches.sort(key=_match_sort_key)
        if self.cache_dir:
            self._cache_event_matches(event_key, matches, raw=data)
        return matches

    def _cache_event_matches(self, event_key: str, matches: list[Match], raw: Any) -> None:
        out_dir = self.cache_dir / event_key
        out_dir.mkdir(parents=True, exist_ok=True)
        # Raw TBA payload (full detail) for forensic use.
        (out_dir / "matches.raw.json").write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
        # Compact form we actually consume downstream.
        compact = [
            {
                "key": m.key,
                "event_key": m.event_key,
                "comp_level": m.comp_level,
                "set_number": m.set_number,
                "match_number": m.match_number,
                "short_name": m.short_name,
                "red_teams": m.red_teams,
                "blue_teams": m.blue_teams,
                "videos": [{"type": v.type, "key": v.key} for v in m.videos],
            }
            for m in matches
        ]
        (out_dir / "matches.json").write_text(
            json.dumps(compact, indent=2), encoding="utf-8"
        )


def _parse_match(raw: dict) -> Match:
    alliances = raw.get("alliances") or {}
    red = (alliances.get("red") or {}).get("team_keys") or []
    blue = (alliances.get("blue") or {}).get("team_keys") or []
    videos = [
        MatchVideo(type=v.get("type", ""), key=v.get("key", ""))
        for v in (raw.get("videos") or [])
    ]
    return Match(
        key=raw["key"],
        event_key=raw["event_key"],
        comp_level=raw["comp_level"],
        set_number=int(raw.get("set_number") or 1),
        match_number=int(raw.get("match_number") or 0),
        red_teams=list(red),
        blue_teams=list(blue),
        videos=videos,
    )


_LEVEL_ORDER = {"pm": 0, "qm": 1, "ef": 2, "qf": 3, "sf": 4, "f": 5}


def _match_sort_key(m: Match) -> tuple[int, int, int]:
    return (_LEVEL_ORDER.get(m.comp_level, 99), m.set_number, m.match_number)


def load_auth_key(secrets_path: Path) -> str:
    """Read the TBA auth key from configs/secrets.yaml."""
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Secrets file not found at {secrets_path}. "
            f"Copy configs/secrets.example.yaml to configs/secrets.yaml and fill it in."
        )
    data = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
    key = data.get("tba_auth_key")
    if not key or key.startswith("YOUR_"):
        raise ValueError(f"tba_auth_key missing or unset in {secrets_path}")
    return key
