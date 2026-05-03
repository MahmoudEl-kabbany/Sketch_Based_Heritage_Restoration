"""Run full batch restoration on all damaged sketches."""
import os
from restoration.pipeline import restore_batch

images = [
    os.path.join("test_images/difficult_test_cases", f)
    for f in sorted(os.listdir("test_images/difficult_test_cases"))
    if f.endswith(".png")
]

results = restore_batch(images)

print("\n\n===== BATCH SUMMARY =====")
total_time = 0
for r in results:
    name = os.path.basename(r.image_path)
    orig_open = sum(1 for p in r.original_paths if not p.is_closed)
    rest_open = sum(1 for p in r.restored_paths if not p.is_closed)
    t = (
        r.report.get("summary", {}).get("processing_time_s")
        if isinstance(r.report, dict)
        else None
    )
    if t is None and isinstance(r.report, dict):
        t = r.report.get("timing_seconds", 0.0)
    if t is None:
        t = 0.0
    b = len(r.bridges)
    print(f"  {name:40s} | {orig_open} open -> {rest_open} open | bridges={b} | {t}s")
    total_time += t
print(f"  Total processing time: {total_time}s")
print("=========================")
