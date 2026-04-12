"""Run full batch restoration on all damaged sketches."""
import os
from restoration.pipeline import restore_batch

images = [
    os.path.join("test_images/damaged_sketches", f)
    for f in sorted(os.listdir("test_images/damaged_sketches"))
    if f.endswith(".png")
]

results = restore_batch(images)

print("\n\n===== BATCH SUMMARY =====")
for r in results:
    name = os.path.basename(r.image_path)
    orig_open = sum(1 for p in r.original_paths if not p.is_closed)
    rest_open = sum(1 for p in r.restored_paths if not p.is_closed)
    t = r.report["timing_seconds"]
    b = len(r.bridges)
    print(f"  {name:40s} | {orig_open} open -> {rest_open} open | bridges={b} | {t}s")
print("=========================")
