"""Bridge fluidity evaluation: MDTW / MJerk of ASP-selected bridges compared with
rejected candidates and with the matching strokes of the undamaged originals.

Usage:
    python evaluate_bridge_vs_original.py              # all matched pairs
    python evaluate_bridge_vs_original.py --limit 10   # first 10 pairs only
    python evaluate_bridge_vs_original.py --workers 1  # one pair at a time
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from evaluate_restoration import discover_pairs
from restoration.extraction import extract_paths
from restoration.candidates import generate_candidates
from restoration.scoring import compute_mdtw, compute_mjerk, context_curve, score_candidates
from restoration.asp_engine import solve_candidates
from restoration.pipeline import default_workers, map_in_processes

# MJerk (mean squared third derivative, px^-4) is reported in units of 1e-3.
MJERK_SCALE = 1e-3

# evaluate_image() result keys: one list of per-curve values each.
RESULT_KEYS = ("chosen_mdtw", "chosen_mjerk", "rejected_mdtw", "rejected_mjerk", "orig_mdtw", "orig_mjerk")


def evaluate_image(damaged_path: str, orig_path: str, tail_len=20):
    """MDTW / MJerk lists (see RESULT_KEYS) for one damaged/original image pair."""
    out = {key: [] for key in RESULT_KEYS}

    dmg_ext = extract_paths(damaged_path)
    if not dmg_ext.paths:
        return out

    candidates = generate_candidates(dmg_ext, lookahead_fraction=0.15, max_per_endpoint=5)
    scored = score_candidates(candidates, dmg_ext)
    accepted_ids, _ = solve_candidates(scored)
    accepted_ids_set = set(accepted_ids)

    # Sample the original's paths and remember which path each sample came from.
    orig_path_samples = [p.sample(pts_per_segment=20) for p in extract_paths(orig_path).paths]
    orig_path_mapping = [
        (p_idx, pt_idx)
        for p_idx, samples in enumerate(orig_path_samples)
        for pt_idx in range(len(samples))
    ]
    if not orig_path_samples:
        return out
    kdtree = cKDTree(np.vstack(orig_path_samples))

    dense_samples = [p.sample(pts_per_segment=50) for p in dmg_ext.paths]
    for c in scored:
        combined_dmg = context_curve(c, dense_samples, tail_len)
        c_mdtw = compute_mdtw(combined_dmg)
        c_mjerk = compute_mjerk(combined_dmg)

        if c.id not in accepted_ids_set:
            out["rejected_mdtw"].append(c_mdtw)
            out["rejected_mjerk"].append(c_mjerk)
            continue

        # Match both bridge endpoints to the same stroke of the original.
        dist_a, idx_a = kdtree.query(c.ep_a.position)
        dist_b, idx_b = kdtree.query(c.ep_b.position)
        if dist_a > 15.0 or dist_b > 15.0:
            continue

        orig_p_idx_a, pt_idx_a = orig_path_mapping[idx_a]
        orig_p_idx_b, pt_idx_b = orig_path_mapping[idx_b]
        if orig_p_idx_a != orig_p_idx_b:
            continue

        # The original stroke between the two matches, with the same context length.
        orig_samples = orig_path_samples[orig_p_idx_a]
        idx_start = max(0, min(pt_idx_a, pt_idx_b) - tail_len)
        idx_end = min(len(orig_samples), max(pt_idx_a, pt_idx_b) + tail_len + 1)
        combined_orig = orig_samples[idx_start:idx_end]
        if pt_idx_a > pt_idx_b:
            combined_orig = combined_orig[::-1]

        out["chosen_mdtw"].append(c_mdtw)
        out["chosen_mjerk"].append(c_mjerk)
        out["orig_mdtw"].append(compute_mdtw(combined_orig))
        out["orig_mjerk"].append(compute_mjerk(combined_orig))

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="Image pairs evaluated in parallel (1 = one at a time).")
    args = parser.parse_args()

    pairs = discover_pairs()
    if args.limit:
        pairs = pairs[:args.limit]

    totals = {key: [] for key in RESULT_KEYS}

    print(f"Starting evaluation on {len(pairs)} image pairs...")

    jobs = [(dmg_p, orig_p) for _, orig_p, dmg_p in pairs]

    def announce(i: int) -> None:
        print(f"[{i+1}/{len(pairs)}] Evaluating {pairs[i][0]}...")

    for result in map_in_processes(evaluate_image, jobs, args.workers, announce):
        print(f"    Found {len(result['chosen_mdtw'])} matched bridges "
              f"and {len(result['rejected_mdtw'])} rejected candidates.")
        for key in RESULT_KEYS:
            totals[key].extend(result[key])

    if not totals["chosen_mdtw"]:
        print("No matched bridges evaluated.")
        return

    print("\n================================================================================")
    print("  FLUIDITY METRICS RESULTS (RESTORED VS GROUND-TRUTH ORIGINAL)")
    print("================================================================================")

    mean = {key: (np.mean(values) if values else 0.0) for key, values in totals.items()}
    mean_dmg_mdtw, mean_dmg_mjerk = mean["chosen_mdtw"], mean["chosen_mjerk"] / MJERK_SCALE
    mean_rej_mdtw, mean_rej_mjerk = mean["rejected_mdtw"], mean["rejected_mjerk"] / MJERK_SCALE
    mean_orig_mdtw, mean_orig_mjerk = mean["orig_mdtw"], mean["orig_mjerk"] / MJERK_SCALE

    print(f"Total matched bridges evaluated: {len(totals['chosen_mdtw'])}")
    print(f"Chosen Bridges (ASP)     : Mean MDTW = {mean_dmg_mdtw:7.2f}, Mean MJerk = {mean_dmg_mjerk:6.2f}e-3")
    print(f"Rejected Candidates      : Mean MDTW = {mean_rej_mdtw:7.2f}, Mean MJerk = {mean_rej_mjerk:6.2f}e-3")
    print(f"Intact Human Orig Strokes: Mean MDTW = {mean_orig_mdtw:7.2f}, Mean MJerk = {mean_orig_mjerk:6.2f}e-3")

    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative comparison of connection fluidity (Mean MJerk) and approach-path directness (Mean MDTW) comparing the ASP-selected bridges against rejected candidates and their exact corresponding non-damaged ground-truth segments from the original unblemished sketches. The scale for MJerk is presented as $\\times 10^{{-3}}$ units. Metrics for both groups evaluate the exact same full approach curves.}}
\\label{{tab:fluidity_metrics}}
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Connection Strategy}} & \\textbf{{Mean MJerk ($\\times 10^{{-3}}$)}} & \\textbf{{Mean MDTW}} \\\\
\\midrule
Intact Human Orig Strokes (Ground Truth) & {mean_orig_mjerk:.2f} & {mean_orig_mdtw:.2f} \\\\
\\textbf{{Chosen Bridges (ASP Selected)}} & \\mathbf{{{mean_dmg_mjerk:.2f}}} & \\mathbf{{{mean_dmg_mdtw:.2f}}} \\\\
Rejected Candidates & {mean_rej_mjerk:.2f} & {mean_rej_mdtw:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(latex)

    # Plot the results as two subplots side-by-side
    categories = ['Ground Truth', 'Chosen Bridges', 'Rejected Candidates']
    # Colors: Ground Truth (grey), Chosen Bridges (green), Rejected (red/orange)
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']
    charts = (
        ('Mean MDTW (Lower is better)', [mean_orig_mdtw, mean_dmg_mdtw, mean_rej_mdtw]),
        ('Mean Mjerk, x1e-3 (Lower is better)', [mean_orig_mjerk, mean_dmg_mjerk, mean_rej_mjerk]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle('Bridge Metrics: Mean MDTW and Mjerk', fontsize=16, fontweight='bold', y=1.05)

    for ax, (chart_title, values) in zip(axes, charts):
        bars = ax.bar(categories, values, color=colors, width=0.6)
        ax.set_title(chart_title, fontsize=13)
        ax.set_ylim(0, max(values) * 1.15)
        ax.grid(axis='y', linestyle='-', alpha=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', rotation=15)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig('bridge_vs_original_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved evaluation graph to 'bridge_vs_original_metrics.png'")


if __name__ == "__main__":
    main()
