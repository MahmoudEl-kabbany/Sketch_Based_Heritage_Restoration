import json

with open('restoration/outputs/occluded_rectangle_arch_report.json', 'r') as f:
    d = json.load(f)

for e in d.get('detailed_events', []):
    if e.get('source') == 'asp_bridge':
        c = e.get('candidate', {})
        print(f"{e['id']}: {c.get('scenario')} tier={c.get('tier')} dist={c.get('distance_px')}")
