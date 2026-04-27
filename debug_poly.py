import json

with open('restoration/outputs/perspective_distorted_polygon_gaps_report.json', 'r') as f:
    d = json.load(f)

for e in d.get('detailed_events', []):
    if e.get('source') == 'asp_bridge':
        c = e.get('candidate', {})
        print(f"{e['id']}: {c.get('scenario')} dist={c.get('distance_px')} score={c.get('score')}")
