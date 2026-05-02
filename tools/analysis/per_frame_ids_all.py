"""Per-frame IDS for ALL val scenes, sorted by total IDS desc.

Compact grid layout: each cell shows IDS bars per frame for one scene.
Color intensity scaled to total IDS (yellow=low, red=high).

Usage:
    python tools/per_frame_ids_all.py --results work_dirs/.../tracking_results.json \
        --out per_frame_ids_all.png
"""
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import motmetrics as mm
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES

TN = {'bicycle','bus','car','motorcycle','pedestrian','trailer','truck'}

def cat2t(c):
    for n in ['pedestrian','car','truck','bus','trailer','motorcycle','bicycle']:
        if n in c: return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--out', default='per_frame_ids_all.png')
    ap.add_argument('--cols', type=int, default=10)
    args = ap.parse_args()

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    pred = json.load(open(args.results))['results']

    # Process all val scenes
    per_scene = []
    for s in nusc.scene:
        if s['name'] not in VAL_SCENES: continue
        acc = mm.MOTAccumulator(auto_id=True)
        samples = []; tok = s['first_sample_token']
        while tok: samples.append(tok); tok = nusc.get('sample', tok)['next']
        n_gt_per = []
        for tok in samples:
            sample = nusc.get('sample', tok)
            gt_ids, gt_pts = [], []
            for ann_tok in sample['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                t = cat2t(ann['category_name'])
                if t:
                    gt_ids.append(ann['instance_token'])
                    gt_pts.append(ann['translation'][:2])
            n_gt_per.append(len(gt_ids))
            pred_ids, pred_pts = [], []
            for d in pred.get(tok, []):
                if d['tracking_name'] not in TN: continue
                pred_ids.append(d['tracking_id'])
                pred_pts.append(d['translation'][:2])
            if gt_pts and pred_pts:
                dists = mm.distances.norm2squared_matrix(
                    np.array(gt_pts), np.array(pred_pts), max_d2=4.0)
            else:
                dists = np.empty((len(gt_pts), len(pred_pts)))
            acc.update(gt_ids, pred_ids, dists)
        events = acc.events
        per_frame = np.zeros(len(samples))
        if 'Type' in events.columns:
            sw = events[events['Type'] == 'SWITCH']
            sc = sw.index.get_level_values(0).value_counts().sort_index()
            for fid, cnt in sc.items():
                if 0 <= fid < len(samples): per_frame[fid] = cnt
        # Also count predictions per frame
        n_pred_per = []
        for tok in samples:
            n_pred_per.append(sum(1 for d in pred.get(tok, [])
                                  if d['tracking_name'] in TN))
        per_scene.append({
            'name': s['name'], 'total_ids': int(per_frame.sum()),
            'per_frame': per_frame, 'n_gt': np.array(n_gt_per),
            'n_pred': np.array(n_pred_per),
        })
    per_scene.sort(key=lambda x: -x['total_ids'])

    n = len(per_scene)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 1.4))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    # Shared scales so cells are visually comparable
    y_ids_max = max(int(s['per_frame'].max()) for s in per_scene) or 1
    y_obj_max = max(int(max(s['n_gt'].max(), s['n_pred'].max()))
                    for s in per_scene) or 1

    for i, sc in enumerate(per_scene):
        ax = axes[i]
        x = np.arange(len(sc['per_frame']))
        ax.bar(x, sc['per_frame'], color='#d62728', edgecolor='black',
               linewidth=0.2, label='IDS')
        ax2 = ax.twinx()
        ax2.plot(x, sc['n_gt'], color='#1f77b4', linewidth=0.7, alpha=0.7,
                 label='#GT')
        ax2.plot(x, sc['n_pred'], color='#2ca02c', linewidth=0.7, alpha=0.7,
                 linestyle='--', label='#Pred')
        ax.set_ylim(0, y_ids_max)
        ax2.set_ylim(0, y_obj_max)
        ax2.set_yticks([])
        ax.set_title(f"{sc['name'].replace('scene-','')} ids={sc['total_ids']}",
                     fontsize=7, pad=2)
        ax.tick_params(axis='both', labelsize=5, length=2, pad=1)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Per-frame IDS — all {n} val scenes (sorted by total IDS desc) | '
                 f'red=IDS, blue=#GT, green-dashed=#Pred',
                 fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'Saved: {args.out}')

if __name__ == '__main__':
    main()
