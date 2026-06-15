"""Summarize a match identity file: per-team track coverage, OCR confidence."""
from __future__ import annotations

import json
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    print("usage: python tools/inspect_identities.py <path/to/identities.json>",
          file=sys.stderr)
    raise SystemExit(2)

data = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(f"match:       {data['match_key']}")
print(f"red teams:   {data['red_teams']}")
print(f"blue teams:  {data['blue_teams']}")
print(f"OCR samples: {data['samples_taken']}")
print()

tracks = data.get("tracks", {})
n_total = len(tracks)
n_named = sum(1 for t in tracks.values() if t.get("team_number"))
print(f"tracks: {n_total} total, {n_named} attributed to a team ({n_named*100//max(n_total,1)}%)")

# Group tracks by team_number
by_team: dict[str | None, list[dict]] = defaultdict(list)
for tid, t in tracks.items():
    by_team[t.get("team_number")].append({"track_id": int(tid), **t})

print("\nattributed teams (sorted by track count):")
for team in sorted(
    (k for k in by_team if k is not None),
    key=lambda k: -len(by_team[k]),
):
    entries = by_team[team]
    alliance = entries[0]["alliance"]
    total_reads = sum(e["n_matched"] for e in entries)
    print(f"  team {team:>5} ({alliance:>4}): {len(entries):>2} track(s), "
          f"{total_reads} confirming reads")

unattributed = by_team.get(None, [])
if unattributed:
    # Break down unattributed by class — were they robots that just never
    # had readable bumpers, or non-robots?
    cls_counts = defaultdict(int)
    for e in unattributed:
        cls_counts[e["class_majority"]] += 1
    print(f"\nunattributed ({len(unattributed)} tracks):")
    for cls, n in sorted(cls_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:15s} {n}")
