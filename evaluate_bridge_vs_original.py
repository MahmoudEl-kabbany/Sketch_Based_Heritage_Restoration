import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.spatial import cKDTree

from restoration.extraction import extract_paths
from restoration.candidates import generate_candidates
from restoration.scoring import score_candidates
from restoration.asp_engine import solve_partitioned

def resample_equidistant(pts, distance=1.0):
    if len(pts) < 2: return pts
    diffs = np.diff(pts, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dist[-1]
    if total_dist <= 0: return pts
    num_points = int(np.ceil(total_dist / distance)) + 1
    new_cum_dist = np.linspace(0, total_dist, num_points)
    new_x = np.interp(new_cum_dist, cum_dist, pts[:, 0])
    new_y = np.interp(new_cum_dist, cum_dist, pts[:, 1])
    return np.column_stack((new_x, new_y))

def calc_mjerk(combined: np.ndarray) -> float:
    if len(combined) < 5: return 0.0
    diffs = np.diff(combined, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    ds[ds < 1e-12] = 1e-12
    vel = diffs / ds[:, np.newaxis]
    if len(vel) < 2: return 0.0
    d_vel = np.diff(vel, axis=0)
    ds2 = ds[:-1]
    ds2[ds2 < 1e-12] = 1e-12
    acc = d_vel / ds2[:, np.newaxis]
    if len(acc) < 2: return 0.0
    d_acc = np.diff(acc, axis=0)
    ds3 = ds2[:-1]
    ds3[ds3 < 1e-12] = 1e-12
    jerk = d_acc / ds3[:, np.newaxis]
    return float(np.mean(np.sum(jerk ** 2, axis=1)))

def calc_mdtw(combined: np.ndarray) -> float:
    if len(combined) < 4: return 0.0
    baseline = np.linspace(combined[0], combined[-1], len(combined))
    dtw_x = dtw.distance(combined[:, 0].astype(np.double), baseline[:, 0].astype(np.double))
    dtw_y = dtw.distance(combined[:, 1].astype(np.double), baseline[:, 1].astype(np.double))
    return float(np.hypot(dtw_x, dtw_y))

def evaluate_image(damaged_path: str, orig_path: str, tail_len=20):
    dmg_ext = extract_paths(damaged_path)
    if not dmg_ext.paths:
        return [], [], [], []
    
    candidates = generate_candidates(dmg_ext, lookahead_fraction=0.15, max_per_endpoint=5)
    scored = score_candidates(candidates, dmg_ext)
    accepted_ids, _ = solve_partitioned(scored, dmg_ext)
    accepted_ids_set = set(accepted_ids)
    accepted = [c for c in scored if c.id in accepted_ids_set]
    rejected = [c for c in scored if c.id not in accepted_ids_set]

    orig_ext = extract_paths(orig_path)
    
    orig_path_samples = []
    orig_path_mapping = []
    for p_idx, p in enumerate(orig_ext.paths):
        samples = p.sample(pts_per_segment=20)
        orig_path_samples.append(samples)
        for pt_idx in range(len(samples)):
            orig_path_mapping.append((p_idx, pt_idx))
            
    if not orig_path_samples:
        return [], [], [], []
        
    all_orig_pts = np.vstack(orig_path_samples)
    kdtree = cKDTree(all_orig_pts)

    dmg_mdtw, dmg_mjerk = [], []
    dmg_mdtw_rej, dmg_mjerk_rej = [], []
    orig_mdtw, orig_mjerk = [], []

    for c in rejected:
        src_path = dmg_ext.paths[c.ep_a.path_index]
        tgt_path = dmg_ext.paths[c.ep_b.path_index]
        
        source_samples = src_path.sample(pts_per_segment=50)
        target_samples = tgt_path.sample(pts_per_segment=50)

        if c.ep_a.end == "back":
            source_tail = source_samples[-tail_len:]
        else:
            source_tail = source_samples[:tail_len][::-1]

        if c.ep_b.end == "front":
            target_head = target_samples[:tail_len]
        else:
            target_head = target_samples[-tail_len:][::-1]
            
        combined_dmg = np.vstack([source_tail, c.bridge_points, target_head])
        
        c_mdtw = calc_mdtw(combined_dmg)
        c_mjerk = calc_mjerk(combined_dmg)
        
        dmg_mdtw_rej.append(c_mdtw)
        dmg_mjerk_rej.append(c_mjerk)

    for c in accepted:
        src_path = dmg_ext.paths[c.ep_a.path_index]
        tgt_path = dmg_ext.paths[c.ep_b.path_index]
        
        source_samples = src_path.sample(pts_per_segment=50)
        target_samples = tgt_path.sample(pts_per_segment=50)

        if c.ep_a.end == "back":
            source_tail = source_samples[-tail_len:]
        else:
            source_tail = source_samples[:tail_len][::-1]

        if c.ep_b.end == "front":
            target_head = target_samples[:tail_len]
        else:
            target_head = target_samples[-tail_len:][::-1]
            
        combined_dmg = np.vstack([source_tail, c.bridge_points, target_head])
        
        c_mdtw = calc_mdtw(combined_dmg)
        c_mjerk = calc_mjerk(combined_dmg)
        
        pt_a = c.ep_a.position
        pt_b = c.ep_b.position
        
        dist_a, idx_a = kdtree.query(pt_a)
        dist_b, idx_b = kdtree.query(pt_b)
        
        if dist_a > 15.0 or dist_b > 15.0:
            continue
            
        orig_p_idx_a, pt_idx_a = orig_path_mapping[idx_a]
        orig_p_idx_b, pt_idx_b = orig_path_mapping[idx_b]
        
        if orig_p_idx_a != orig_p_idx_b:
            continue
            
        orig_samples = orig_path_samples[orig_p_idx_a]
        
        idx_start = min(pt_idx_a, pt_idx_b)
        idx_end = max(pt_idx_a, pt_idx_b)
        
        idx_start_tail = max(0, idx_start - tail_len)
        idx_end_tail = min(len(orig_samples), idx_end + tail_len + 1)
        
        combined_orig = orig_samples[idx_start_tail:idx_end_tail]
        
        if pt_idx_a > pt_idx_b:
            combined_orig = combined_orig[::-1]
            
        o_mdtw = calc_mdtw(combined_orig)
        o_mjerk = calc_mjerk(combined_orig)
        
        dmg_mdtw.append(c_mdtw)
        dmg_mjerk.append(c_mjerk)
        orig_mdtw.append(o_mdtw)
        orig_mjerk.append(o_mjerk)
        
    return dmg_mdtw, dmg_mjerk, dmg_mdtw_rej, dmg_mjerk_rej, orig_mdtw, orig_mjerk

def discover_pairs():
    orig_dir = os.path.join("test_images", "difficult_test_cases_original")
    dmg_dir = os.path.join("test_images", "difficult_test_cases")
    pairs = []
    origs = {}
    if not os.path.isdir(orig_dir):
        return []
    for f in os.listdir(orig_dir):
        base, ext = os.path.splitext(f)
        origs[base] = os.path.join(orig_dir, f)
        
    for f in os.listdir(dmg_dir):
        base, ext = os.path.splitext(f)
        if base.endswith("_damaged"):
            true_base = base.replace("_damaged", "")
            if true_base in origs:
                pairs.append((true_base, os.path.join(dmg_dir, f), origs[true_base]))
    return sorted(pairs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    pairs = discover_pairs()
    if args.limit:
        pairs = pairs[:args.limit]
        
    all_dmg_mdtw, all_dmg_mjerk = [], []
    all_dmg_mdtw_rej, all_dmg_mjerk_rej = [], []
    all_orig_mdtw, all_orig_mjerk = [], []
    
    print(f"Starting evaluation on {len(pairs)} image pairs...")
    
    for i, (name, dmg_p, orig_p) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] Evaluating {name}...")
        dm, dj, dmr, djr, om, oj = evaluate_image(dmg_p, orig_p)
        print(f"    Found {len(dm)} matched bridges and {len(dmr)} rejected candidates.")
        all_dmg_mdtw.extend(dm)
        all_dmg_mjerk.extend(dj)
        all_dmg_mdtw_rej.extend(dmr)
        all_dmg_mjerk_rej.extend(djr)
        all_orig_mdtw.extend(om)
        all_orig_mjerk.extend(oj)
        
    if not all_dmg_mdtw:
        print("No matched bridges evaluated.")
        return
        
    print("\\n================================================================================")
    print("  FLUIDITY METRICS RESULTS (RESTORED VS GROUND-TRUTH ORIGINAL)")
    print("================================================================================")
    
    mean_dmg_mdtw = np.mean(all_dmg_mdtw)
    mean_dmg_mjerk = np.mean(all_dmg_mjerk)
    mean_dmg_mdtw_rej = np.mean(all_dmg_mdtw_rej) if all_dmg_mdtw_rej else 0.0
    mean_dmg_mjerk_rej = np.mean(all_dmg_mjerk_rej) if all_dmg_mjerk_rej else 0.0
    mean_orig_mdtw = np.mean(all_orig_mdtw)
    mean_orig_mjerk = np.mean(all_orig_mjerk)
    
    mjerk_scale = 1e46
    
    print(f"Total matched bridges evaluated: {len(all_dmg_mdtw)}")
    print(f"Chosen Bridges (ASP)     : Mean MDTW = {mean_dmg_mdtw:7.2f}, Mean MJerk = {mean_dmg_mjerk/mjerk_scale:6.2f}e46")
    print(f"Rejected Candidates      : Mean MDTW = {mean_dmg_mdtw_rej:7.2f}, Mean MJerk = {mean_dmg_mjerk_rej/mjerk_scale:6.2f}e46")
    print(f"Intact Human Orig Strokes: Mean MDTW = {mean_orig_mdtw:7.2f}, Mean MJerk = {mean_orig_mjerk/mjerk_scale:6.2f}e46")
    
    latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Quantitative comparison of connection fluidity (Mean MJerk) and approach-path directness (Mean MDTW) comparing the ASP-selected bridges against rejected candidates and their exact corresponding non-damaged ground-truth segments from the original unblemished sketches. The scale for MJerk is presented as $\\times 10^{{46}}$ units. Metrics for both groups evaluate the exact same full approach curves.}}
\\label{{tab:fluidity_metrics}}
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Connection Strategy}} & \\textbf{{Mean MJerk ($\\times 10^{{46}}$)}} & \\textbf{{Mean MDTW}} \\\\
\\midrule
Intact Human Orig Strokes (Ground Truth) & {mean_orig_mjerk/mjerk_scale:.2f} & {mean_orig_mdtw:.2f} \\\\
\\textbf{{Chosen Bridges (ASP Selected)}} & \\mathbf{{{mean_dmg_mjerk/mjerk_scale:.2f}}} & \\mathbf{{{mean_dmg_mdtw:.2f}}} \\\\
Rejected Candidates & {mean_dmg_mjerk_rej/mjerk_scale:.2f} & {mean_dmg_mdtw_rej:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    print(latex)
    # Plotting the results as two separate subplots side-by-side
    categories = ['Ground Truth', 'Chosen Bridges', 'Rejected Candidates']
    mjerk_values = [mean_orig_mjerk/mjerk_scale, mean_dmg_mjerk/mjerk_scale, mean_dmg_mjerk_rej/mjerk_scale]
    mdtw_values = [mean_orig_mdtw, mean_dmg_mdtw, mean_dmg_mdtw_rej]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.suptitle('Bridge Metrics: Mean MDTW and Mjerk', fontsize=16, fontweight='bold', y=1.05)

    # Colors: Ground Truth (grey), Chosen Bridges (green), Rejected (red/orange)
    colors = ['#95a5a6', '#2ecc71', '#e74c3c']

    # Subplot 1: MDTW
    bars1 = ax1.bar(categories, mdtw_values, color=colors, width=0.6)
    ax1.set_title('Mean MDTW (Lower is better)', fontsize=13)
    ax1.set_ylim(0, max(mdtw_values) * 1.15)
    ax1.grid(axis='y', linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='x', rotation=15)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2),
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Subplot 2: Mjerk
    bars2 = ax2.bar(categories, mjerk_values, color=colors, width=0.6)
    ax2.set_title('Mean Mjerk (Lower is better)', fontsize=13)
    ax2.set_ylim(0, max(mjerk_values) * 1.15)
    ax2.grid(axis='y', linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='x', rotation=15)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 2),
                     textcoords="offset points",
                     ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig('bridge_vs_original_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved evaluation graph to 'bridge_vs_original_metrics.png'")

if __name__ == "__main__":
    main()
