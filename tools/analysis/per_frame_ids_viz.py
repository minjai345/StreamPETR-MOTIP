"""Per-frame IDS distribution for worst scenes.

For each scene, extracts SWITCH events from motmetrics and plots
per-frame IDS counts. Helps diagnose whether ID switches are:
  - Clustered  → exposure bias / corrupted context after one mistake
  - Uniform    → detection/matching threshold issue

Usage:
    python tools/per_frame_ids_viz.py --results work_dirs/.../tracking_results.json \
        --scenes scene-0105,scene-0276,scene-0018 --out per_frame_ids.png
"""
import argparse, json, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import motmetrics as mm
from nuscenes import NuScenes

TRACKING_NAMES = {'bicycle','bus','car','motorcycle','pedestrian','trailer','truck'}

def cat_to_tname(cat):
    for n in ['pedestrian','car','truck','bus','trailer','motorcycle','bicycle']:
        if n in cat: return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--scenes', required=True,
                    help='Comma-separated scene names (e.g., scene-0105,scene-0276)')
    ap.add_argument('--out', default='per_frame_ids.png')
    args = ap.parse_args()

    target_scenes = args.scenes.split(',')
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    pred = json.load(open(args.results))['results']

    fig, axes = plt.subplots(len(target_scenes), 1, figsize=(12, 2.2 * len(target_scenes)))
    if len(target_scenes) == 1: axes = [axes]

    for ax_i, sname in enumerate(target_scenes):
        scene = next((s for s in nusc.scene if s['name'] == sname), None)
        if scene is None:
            print(f'Skip {sname}: not found'); continue

        # Iterate frames and accumulate
        acc = mm.MOTAccumulator(auto_id=True)
        n_gt_per_frame, n_pred_per_frame = [], []
        tok = scene['first_sample_token']
        while tok:
            sample = nusc.get('sample', tok)
            gt_ids, gt_pts = [], []
            for ann_tok in sample['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                tname = cat_to_tname(ann['category_name'])
                if tname:
                    gt_ids.append(ann['instance_token'])
                    gt_pts.append(ann['translation'][:2])
            pred_ids, pred_pts = [], []
            for d in pred.get(tok, []):
                if d['tracking_name'] not in TRACKING_NAMES: continue
                pred_ids.append(d['tracking_id'])
                pred_pts.append(d['translation'][:2])
            n_gt_per_frame.append(len(gt_ids))
            n_pred_per_frame.append(len(pred_ids))

            if gt_pts and pred_pts:
                dists = mm.distances.norm2squared_matrix(
                    np.array(gt_pts), np.array(pred_pts), max_d2=4.0)
            else:
                dists = np.empty((len(gt_pts), len(pred_pts)))
            acc.update(gt_ids, pred_ids, dists)
            tok = sample['next']

        # Extract per-frame SWITCH count from events
        events = acc.events
        if 'Type' in events.columns:
            switches = events[events['Type'] == 'SWITCH']
            # Frame index is the first level of MultiIndex
            switch_counts = switches.index.get_level_values(0).value_counts().sort_index()
        else:
            switch_counts = defaultdict(int)
        n_frames = len(n_gt_per_frame)
        per_frame = np.zeros(n_frames)
        for fid, cnt in switch_counts.items():
            if 0 <= fid < n_frames:
                per_frame[fid] = cnt
        total = int(per_frame.sum())

        ax = axes[ax_i]
        ax2 = ax.twinx()
        x = np.arange(n_frames)
        ax.bar(x, per_frame, color='#d62728', alpha=0.85, label='IDS')
        ax2.plot(x, n_gt_per_frame, color='#1f77b4', linewidth=1.2, alpha=0.7, label='#GT')
        ax2.plot(x, n_pred_per_frame, color='#2ca02c', linewidth=1.2,
                 alpha=0.7, linestyle='--', label='#Pred')
        ax.set_title(f'{sname}  (total IDS={total}, frames={n_frames})')
        ax.set_xlabel('frame index'); ax.set_ylabel('IDS', color='#d62728')
        ax2.set_ylabel('#objects', color='#1f77b4')
        ax.set_xlim(-0.5, n_frames - 0.5)
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'Saved: {args.out}')

if __name__ == '__main__':
    main()
