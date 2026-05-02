"""Compute per-scene IDS for a tracking_results.json against nuScenes GT.

Uses motmetrics with 2m center-distance matching (close to nuScenes default
2.0m threshold per class). Aggregates across all tracking classes per scene.

Usage:
    python tools/per_scene_ids.py --results work_dirs/.../tracking_results.json \
        [--top 20]
"""
import argparse, json
from collections import defaultdict
import numpy as np
import motmetrics as mm
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES

TRACKING_NAMES = {'bicycle', 'bus', 'car', 'motorcycle',
                  'pedestrian', 'trailer', 'truck'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--top', type=int, default=20,
                    help='Show top-N worst scenes by IDS')
    args = ap.parse_args()

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    pred = json.load(open(args.results))['results']

    val_scenes = set(VAL_SCENES)
    sample_to_scene = {}
    scene_to_samples = defaultdict(list)
    scene_name = {}
    for s in nusc.scene:
        if s['name'] not in val_scenes:
            continue
        scene_name[s['token']] = s['name']
        token = s['first_sample_token']
        while token:
            sample_to_scene[token] = s['token']
            scene_to_samples[s['token']].append(token)
            token = nusc.get('sample', token)['next']

    per_scene_ids = {}
    per_scene_mota = {}

    for scene_tok, samples in scene_to_samples.items():
        acc = mm.MOTAccumulator(auto_id=True)
        for tok in samples:
            sample = nusc.get('sample', tok)
            # GT: instance_token + LiDAR-frame translation (any tracking class)
            gt_ids, gt_pts = [], []
            for ann_tok in sample['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                cat = ann['category_name']
                # nuScenes raw category to tracking name (simplified)
                if 'car' in cat: tname = 'car'
                elif 'pedestrian' in cat: tname = 'pedestrian'
                elif 'truck' in cat: tname = 'truck'
                elif 'bus' in cat: tname = 'bus'
                elif 'trailer' in cat: tname = 'trailer'
                elif 'motorcycle' in cat: tname = 'motorcycle'
                elif 'bicycle' in cat: tname = 'bicycle'
                else: continue
                if tname not in TRACKING_NAMES: continue
                gt_ids.append(ann['instance_token'])
                gt_pts.append(ann['translation'][:2])

            # Pred: tracking_id + global translation
            pred_ids, pred_pts = [], []
            for d in pred.get(tok, []):
                if d['tracking_name'] not in TRACKING_NAMES: continue
                pred_ids.append(d['tracking_id'])
                pred_pts.append(d['translation'][:2])

            if gt_pts and pred_pts:
                dists = mm.distances.norm2squared_matrix(
                    np.array(gt_pts), np.array(pred_pts), max_d2=4.0)
            else:
                dists = np.empty((len(gt_pts), len(pred_pts)))
            acc.update(gt_ids, pred_ids, dists)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_switches', 'mota', 'num_objects'],
                             name='s')
        per_scene_ids[scene_tok] = int(summary.loc['s', 'num_switches'])
        per_scene_mota[scene_tok] = float(summary.loc['s', 'mota'])

    # Sort by IDS desc
    items = sorted(per_scene_ids.items(), key=lambda x: -x[1])
    total = sum(per_scene_ids.values())
    print(f'Total scenes: {len(per_scene_ids)}, Total IDS: {total}')
    print(f'Mean IDS/scene: {total / len(per_scene_ids):.1f}')
    print()
    print(f'Top {args.top} worst scenes by IDS:')
    print(f'{"scene_name":<14} {"IDS":>5} {"MOTA":>7}')
    for tok, ids in items[:args.top]:
        print(f'{scene_name[tok]:<14} {ids:>5d} {per_scene_mota[tok]:>7.3f}')

if __name__ == '__main__':
    main()
