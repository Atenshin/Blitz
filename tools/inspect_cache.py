"""Summarize a detection cache file: schema version, track stats, sample tracks.

Usage:
    python tools/inspect_cache.py detections/2026cmptx/2026cmptx_sf1m1.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter

if len(sys.argv) < 2:
    print("usage: python tools/inspect_cache.py <path/to/cache.json>", file=sys.stderr)
    raise SystemExit(2)

data = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"schema_version: {data['schema_version']}")
print(f"tracking_used:  {data.get('tracking_used')}")
print(f"tracker:        {data.get('tracker')}")
print(f"frames:         {len(data['frames'])}")

total = with_id = 0
unique_ids: set[int] = set()
unique_ids_by_class: dict[str, set[int]] = {}
for f in data['frames']:
    for d in f['detections']:
        total += 1
        if d.get('object_id') is not None:
            with_id += 1
            unique_ids.add(d['object_id'])
            unique_ids_by_class.setdefault(d['name'], set()).add(d['object_id'])

print(f"detections:     {total} total, {with_id} have track IDs ({with_id*100//total}%)")
print(f"unique track IDs total: {len(unique_ids)}")
print(f"unique track IDs per class:")
for name, ids in sorted(unique_ids_by_class.items(), key=lambda x: -len(x[1])):
    print(f"  {name:15s} {len(ids)} distinct IDs")

# How long does an average track live?
first_seen: dict[int, float] = {}
last_seen: dict[int, float] = {}
for f in data['frames']:
    for d in f['detections']:
        oid = d.get('object_id')
        if oid is None:
            continue
        first_seen.setdefault(oid, f['sec'])
        last_seen[oid] = f['sec']

durations = [last_seen[i] - first_seen[i] for i in unique_ids]
durations.sort()
if durations:
    mid = durations[len(durations) // 2]
    print(f"\ntrack lifetime distribution:")
    print(f"  median:  {mid:.1f}s")
    print(f"  longest: {durations[-1]:.1f}s")
    long_tracks = sum(1 for d in durations if d > 10.0)
    print(f"  tracks lasting > 10s: {long_tracks} / {len(durations)}")

# Show a few sample tracked robots end-to-end
robot_ids = sorted(unique_ids_by_class.get('robot_blue', set()) |
                   unique_ids_by_class.get('robot_red', set()))[:5]
print(f"\nsample robot tracks:")
for oid in robot_ids:
    cls_in_frames = []
    for f in data['frames']:
        for d in f['detections']:
            if d.get('object_id') == oid:
                cls_in_frames.append((f['sec'], d['name']))
                break
    if cls_in_frames:
        first_t = cls_in_frames[0][0]
        last_t = cls_in_frames[-1][0]
        cls = cls_in_frames[0][1]
        n = len(cls_in_frames)
        print(f"  #{oid:4d} {cls:11s} {first_t:6.1f}s -> {last_t:6.1f}s  "
              f"({last_t - first_t:5.1f}s, {n} appearances)")
