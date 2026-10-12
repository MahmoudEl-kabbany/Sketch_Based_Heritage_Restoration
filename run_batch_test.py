"""Run full batch restoration on all damaged sketches."""
import argparse
import os
import sys
from restoration.pipeline import default_workers, restore_batch


def main():
    parser = argparse.ArgumentParser(description="Restore every damaged sketch of the test dataset.")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Images restored in parallel (1 = one at a time).")
    args = parser.parse_args()

    dataset_dir = os.path.join("test_images", "difficult_test_cases")
    if not os.path.isdir(dataset_dir):
        print(f"\nError: Dataset directory not found at '{dataset_dir}'.")
        print("Please download the test dataset from the link in the README and extract it correctly.")
        sys.exit(1)

    images = [
        os.path.join(dataset_dir, f)
        for f in sorted(os.listdir(dataset_dir))
        if f.endswith(".png")
    ]

    results = restore_batch(images, workers=args.workers)

    print("\n\n===== BATCH SUMMARY =====")
    total_time = 0
    for r in results:
        name = os.path.basename(r.image_path)
        orig_open = sum(1 for p in r.original_paths if not p.is_closed)
        rest_open = sum(1 for p in r.restored_paths if not p.is_closed)
        t = r.report["summary"]["processing_time_s"]
        b = len(r.bridges)
        print(f"  {name:40s} | {orig_open} open -> {rest_open} open | bridges={b} | {t}s")
        total_time += t
    print(f"  Total processing time: {total_time}s")
    print("=========================")


# The file name matches pytest's "*_test.py" pattern, so the batch must not
# start when the module is merely imported.
if __name__ == "__main__":
    main()
