import json

with open('restoration/outputs/damaged_eye_report.json', 'r') as f:
    d = json.load(f)

print('Open paths remaining:', d['summary']['open_paths_after'])
for e in d.get('detailed_events', []):
    if e.get('source') == 'asp_bridge':
        c = e.get('candidate', {})
        print(f"{e['id']}: {c.get('scenario')} dist={c.get('distance_px')} tier={c.get('tier')} ep_a={c.get('endpoint_a')} ep_b={c.get('endpoint_b')}")

with open('restoration/outputs/difficult_fragmented_spiral_report.json', 'r') as f:
    d = json.load(f)
print('\nSpiral open paths:', d['summary']['open_paths_after'])
