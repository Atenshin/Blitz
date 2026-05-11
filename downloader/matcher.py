"""Parse FRC match identifiers out of free-form video titles.

Handles common conventions seen on the FIRSTinMichigan, FRC Game Day, and
The Blue Alliance YouTube channels, as well as raw event-broadcast titles.

Output format mirrors TBA: "<event_key>_<comp_level><set>m<match>" for playoffs
and "<event_key>_qm<n>" / "_pm<n>" / "_f1m<n>" for the rest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: try playoff patterns before plain "match N" so "Quarterfinal 1
# Match 2" doesn't grab the trailing 2 as the qual number.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("qm", re.compile(r"\b(?:qualification|qualif|qual|q)\s*[#:.\-]?\s*(\d+)\b", re.I)),
    ("pm", re.compile(r"\b(?:practice|prac|p)\s*[#:.\-]?\s*(\d+)\b", re.I)),
    # Playoff double-elim form (2023+): "Playoff 5", "Playoff Match 5", "PO 5"
    ("sf", re.compile(r"\b(?:playoff|po)\s*(?:match\s*)?(\d+)\b", re.I)),
    # Quarterfinal/Semifinal with explicit set + match: "QF1-2", "SF2 Match 1"
    ("qf", re.compile(r"\b(?:quarterfinals?|qf)\s*(\d+)\s*[-x:]?\s*(?:match\s*)?(\d+)\b", re.I)),
    ("sf", re.compile(r"\b(?:semifinals?|sf)\s*(\d+)\s*[-x:]?\s*(?:match\s*)?(\d+)\b", re.I)),
    # Eighth-finals (legacy)
    ("ef", re.compile(r"\b(?:eighth\-?finals?|ef)\s*(\d+)\s*[-x:]?\s*(?:match\s*)?(\d+)\b", re.I)),
    # Finals compact form: "F1-2", "F1M2"  (set is always 1; second number is match)
    ("f_compact", re.compile(r"\bf\s*1\s*[-mx:.]\s*(\d+)\b", re.I)),
    # Finals: "Final 1", "Finals Match 2", "Final Match 3"
    ("f",  re.compile(r"\b(?:finals?|f)\s*[#:.\-]?\s*(?:match\s*)?(\d+)\b", re.I)),
]


@dataclass
class ParsedMatch:
    comp_level: str   # qm | qf | sf | ef | f | pm
    set_number: int
    match_number: int

    def key(self, event_key: str) -> str:
        if self.comp_level in ("qm", "pm"):
            return f"{event_key}_{self.comp_level}{self.match_number}"
        return f"{event_key}_{self.comp_level}{self.set_number}m{self.match_number}"


def parse_title(title: str) -> ParsedMatch | None:
    """Return a ParsedMatch, or None if no recognizable pattern was found.

    The first matching pattern wins. Patterns are ordered so that more specific
    forms (playoff/qf/sf/f) are tried before plain "Q N".
    """
    if not title:
        return None

    # Strip parenthetical noise like "(Live)" that can confuse the regex.
    cleaned = re.sub(r"\([^)]*\)", " ", title)

    # Try playoff patterns first (they're more specific).
    for level, pattern in _PATTERNS:
        m = pattern.search(cleaned)
        if not m:
            continue
        groups = m.groups()
        if level in ("qm", "pm"):
            return ParsedMatch(level, 1, int(groups[0]))
        if level == "sf" and len(groups) == 1:
            # 2023+ double-elim "Playoff N" form: set N, match 1
            return ParsedMatch("sf", int(groups[0]), 1)
        if level == "f_compact":
            return ParsedMatch("f", 1, int(groups[0]))
        if level == "f":
            # Finals are always set 1, match N
            return ParsedMatch("f", 1, int(groups[0]))
        # qf / sf / ef with explicit set + match
        return ParsedMatch(level, int(groups[0]), int(groups[1]))

    return None
