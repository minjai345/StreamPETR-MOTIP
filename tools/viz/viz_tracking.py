"""Visualize tracking results vs GT for one scene (BEV top-down)."""
import json, pickle, argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', required=True)
    parser.add_argument('--scene', default=None, help='Scene name (e.g. scene-0003)')
    parser.add_argument('--frame', type=int, default=0, help='Frame index in scene')
    parser.add_argument('--out', default='viz_tracking.png')
    parser.add_argument('--dataroot', default='data/nuscenes/')
    args = parser.parse_args()

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    # Load predictions and GT
    cfg = config_factory('tracking_nips_2019')
    pred_boxes, _ = load_prediction(args.submission, 500, TrackingBox)
    gt_boxes = load_gt(nusc, 'val', TrackingBox)

    # Find scene
    val_scenes = {s['name']: s for s in nusc.scene if s['name'] in set(VAL_SCENES)}
    if args.scene is None:
        args.scene = list(val_scenes.keys())[0]
    scene = val_scenes[args.scene]

    # Get scene sample tokens in order
    tokens = []
    tok = scene['first_sample_token']
    while tok:
        tokens.append(tok)
        sample = nusc.get('sample', tok)
        tok = sample['next']

    frame_tok = tokens[args.frame]
    preds = pred_boxes[frame_tok] if frame_tok in pred_boxes.sample_tokens else []
    gts = gt_boxes[frame_tok] if frame_tok in gt_boxes.sample_tokens else []

    print(f"Scene: {args.scene}, Frame: {args.frame}, Token: {frame_tok}")
    print(f"  Predictions: {len(preds)}, GT: {len(gts)}")

    # Get ego pose for this sample
    sample = nusc.get('sample', frame_tok)
    sd = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego = nusc.get('ego_pose', sd['ego_pose_token'])
    ego_pos = np.array(ego['translation'][:2])

    # Plot BEV
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax_idx, (title, boxes, color) in enumerate([
        ('GT', gts, 'green'),
        ('Predictions', preds, 'red'),
        ('Overlay', None, None),
    ]):
        ax = axes[ax_idx]
        ax.set_title(f'{title} ({args.scene} frame {args.frame})')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Plot ego
        ax.plot(ego_pos[0], ego_pos[1], 'b*', markersize=15, label='ego')

        if ax_idx < 2:
            for b in boxes:
                pos = np.array(b.translation[:2])
                ax.plot(pos[0], pos[1], 'o', color=color, markersize=4, alpha=0.7)
                if hasattr(b, 'tracking_id'):
                    ax.annotate(f'{b.tracking_name[:3]}', pos, fontsize=5, alpha=0.5)
        else:
            # Overlay
            for b in gts:
                pos = np.array(b.translation[:2])
                ax.plot(pos[0], pos[1], 'o', color='green', markersize=6, alpha=0.5, label='GT' if b == gts[0] else '')
            for b in preds:
                pos = np.array(b.translation[:2])
                ax.plot(pos[0], pos[1], 'x', color='red', markersize=5, alpha=0.7, label='Pred' if b == preds[0] else '')
            ax.legend()

        # Set range around ego
        r = 80
        ax.set_xlim(ego_pos[0] - r, ego_pos[0] + r)
        ax.set_ylim(ego_pos[1] - r, ego_pos[1] + r)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

    # Also print distances
    if preds and gts:
        print("\nClosest pred-GT pairs:")
        for p in preds[:10]:
            same = [g for g in gts if g.tracking_name == p.tracking_name]
            if same:
                dists = [(np.linalg.norm(np.array(p.translation[:2]) - np.array(g.translation[:2])), g) for g in same]
                d, g = min(dists, key=lambda x: x[0])
                print(f"  {p.tracking_name}: pred={np.round(p.translation[:2],1)} gt={np.round(g.translation[:2],1)} dist={d:.1f}m id={p.tracking_id}")


if __name__ == '__main__':
    main()
