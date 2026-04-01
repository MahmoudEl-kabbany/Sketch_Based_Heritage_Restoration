"""
Run the restoration pipeline on all test images.
=================================================
Usage:
    python run_restoration_tests.py
"""

from __future__ import annotations

import os
import sys
import traceback

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from restoration.pipeline import restore_image, PipelineConfig

OUTPUT_DIR = os.path.join(_ROOT, "restoration_outputs")
TEST_IMAGES_DIR = os.path.join(_ROOT, "test_images")

TEST_CASES = [
    {
        "file": "restoration_test_damaged.png",
        "original": "restoration_test_original.png",
        "description": "5 shapes with damage (compared to original)",
    },
    {
        "file": "restoration_test.png",
        "original": None,
        "description": "Multiple shapes — no original for comparison",
    },
    {
        "file": "restoration_small_gaps.png",
        "original": None,
        "description": "Diamond/rhombus with two small gaps",
    },
    {
        "file": "damaged_shape.png",
        "original": None,
        "description": "Pentagon with one small gap",
    },
    {
        "file": "damaged_oval.png",
        "original": None,
        "description": "Oval with gap at top",
    },
]


def main():
    print("=" * 70)
    print("  SKETCH-BASED HERITAGE RESTORATION - Test Suite")
    print("=" * 70)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = PipelineConfig()

    passed = 0
    failed = 0
    results_summary = []

    for tc in TEST_CASES:
        image_path = os.path.join(TEST_IMAGES_DIR, tc["file"])
        if not os.path.isfile(image_path):
            print(f"  SKIP: {tc['file']} - file not found")
            continue

        print(f"\n{'-' * 60}")
        print(f"  TEST: {tc['file']}")
        print(f"  {tc['description']}")
        print(f"{'-' * 60}")

        try:
            result = restore_image(image_path, OUTPUT_DIR, config)

            n_bridges = len(result.restoration.new_segments)
            n_arcs = len(result.restoration.efd_arcs)
            n_paths_created = len(result.restoration.new_paths)
            total = n_bridges + n_arcs + n_paths_created

            summary = {
                "file": tc["file"],
                "status": "PASS",
                "paths": len(result.paths),
                "gaps_found": len(result.bundle.gaps),
                "closure_candidates": len(result.bundle.closure_candidates),
                "bridges": n_bridges,
                "efd_arcs": n_arcs,
                "new_paths": n_paths_created,
                "total_restorations": total,
                "timing": sum(result.timing.values()),
            }
            results_summary.append(summary)
            passed += 1

        except Exception as exc:
            print(f"\n  FAILED: {exc}")
            traceback.print_exc()
            results_summary.append({
                "file": tc["file"],
                "status": "FAIL",
                "error": str(exc),
            })
            failed += 1

    # Print summary table
    print(f"\n\n{'=' * 70}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Image':<35} {'Status':<8} {'Gaps':<6} {'Bridges':<9} {'EFD':<6} {'Time':<8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 6} {'-' * 9} {'-' * 6} {'-' * 8}")

    for s in results_summary:
        if s["status"] == "PASS":
            print(f"  {s['file']:<35} {'PASS':<8} "
                  f"{s['gaps_found']:<6} {s['bridges']:<9} "
                  f"{s['efd_arcs']:<6} {s['timing']:.2f}s")
        else:
            print(f"  {s['file']:<35} {'FAIL':<8} {s.get('error', '')[:40]}")

    print(f"\n  Passed: {passed}/{passed + failed}")
    if failed:
        print(f"  Failed: {failed}")
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"{'=' * 70}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
