"""Visualize GT-override effect: BEV plot comparing Normal vs GT-fed tracker
vs GT for a specific scene.

Output: grid of BEV subplots per frame showing:
  - GT boxes (colored by instance_token)
  - Normal tracker (colored by tracking_id)
  - GT-fed tracker (colored by tracking_id)
"""
import os, sys, pickle, hashlib
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes

from tools.eval_tracking import MOTIPTracker, lidar_to_global
from tools.experiments.gt_feeding_eval import GTFeedingTracker
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
from pyquaternion import Quaternion as Q


CFG = "projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e_phase2.py"
CKPT = "work_dirs/motip_e2e_v2_fresh/iter_28128.pth"
FEATS = "work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl"
SCENE = "scene-0105"
FRAME_INDICES = [0, 5, 10, 15, 20, 25, 30, 35]
OUT = "work_dirs/motip_e2e_v2_fresh/gt_feeding/viz_override.png"


def hash_color(key, saturation=0.8, value=0.9):
    """Deterministic color from string key."""
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 360
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360, saturation, value)
    return (r, g, b)


def draw_box(ax, cx, cy, w, l, yaw, color, lw=1.5, alpha=0.9):
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    world = (R @ corners.T).T + np.array([cx, cy])
    poly = plt.Polygon(world, fill=False, edgecolor=color, linewidth=lw, alpha=alpha)
    ax.add_patch(poly)


def setup_model():
    cfg = Config.fromfile(CFG)
    import importlib; importlib.import_module('projects.mmdet3d_plugin')
    model = build_model(cfg.model)
    wrap_fp16_model(model)
    model.cuda().eval()
    load_checkpoint(model, CKPT, map_location='cuda', strict=False)
    return cfg, model


def load_all_feats(cfg):
    raw = pickle.load(open(FEATS, 'rb'))
    val_cfg = cfg.data.val.copy(); val_cfg['test_mode'] = True
    dataset = build_dataset(val_cfg)
    data_infos = dataset.data_infos
    pc_range = [-51.2,-51.2,-5,51.2,51.2,3]
    all_feats = {}
    for idx, out in enumerate(raw):
        info = data_infos[idx]; tok = info['token']
        pts = out['pts_bbox']
        cls_sig = pts['cls_scores'].sigmoid()
        scores_flat, indexs = cls_sig.view(-1).topk(300)
        labels = indexs % 10
        qi = torch.div(indexs, 10, rounding_mode='trunc')
        decoded = denormalize_bbox(pts['bbox_raw'][qi], pc_range)
        l2e_r = Q(info['lidar2ego_rotation']).rotation_matrix
        l2e_t = np.array(info['lidar2ego_translation'])
        e2g_r = Q(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'])
        l2g = np.eye(4); l2g[:3,:3] = e2g_r @ l2e_r
        l2g[:3,3] = e2g_r @ l2e_t + e2g_t
        all_feats[tok] = {
            'scores': scores_flat, 'labels': labels,
            'bbox_decoded': decoded, 'bbox_raw': pts['bbox_raw'][qi],
            'query_feat': pts['query_feat'][qi],
            'ego_pose': torch.from_numpy(l2g).float(),
        }
    return all_feats


def run_trackers(cfg, model, all_feats, scene_samples, nusc, token_to_info):
    mc = dict(cfg.model.motip_cfg)
    mc.update({'det_thresh':0.25,'new_thresh':0.40,'id_thresh':0.10,'max_age':5})

    # Normal
    normal = MOTIPTracker(model, mc); normal.reset()
    normal_out = []
    for tok in scene_samples:
        fd = all_feats[tok]
        normal_out.append(normal.track_frame(fd, fd['ego_pose']))

    # GT-fed
    gtfed = GTFeedingTracker(model, mc); gtfed.reset(); gtfed.reset_scene_gt()
    gtfed_out = []
    for tok in scene_samples:
        fd = all_feats[tok]
        sample = nusc.get('sample', tok)
        gt_inst, gt_centers = [], []
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            cat = ann['category_name']
            if any(n in cat for n in ['pedestrian','car','truck','bus','trailer','motorcycle','bicycle']):
                gt_inst.append(ann['instance_token'])
                gt_centers.append(ann['translation'][:2])
        gtfed.set_frame_gt(gt_inst, gt_centers)
        gtfed._frame_info = token_to_info[tok]
        gtfed_out.append(gtfed.track_frame(fd, fd['ego_pose']))
    return normal_out, gtfed_out


def main():
    cfg, model = setup_model()
    all_feats = load_all_feats(cfg)
    val_data = pickle.load(open(cfg.data.val.ann_file, 'rb'))
    token_to_info = {info['token']: info for info in val_data['infos']}
    nusc = NuScenes(version='v1.0-trainval', dataroot=cfg.data.val.data_root, verbose=False)

    scene = next(s for s in nusc.scene if s['name'] == SCENE)
    samples = []; tok = scene['first_sample_token']
    while tok: samples.append(tok); tok = nusc.get('sample', tok)['next']

    normal_out, gtfed_out = run_trackers(cfg, model, all_feats, samples, nusc, token_to_info)

    # Plot: 3 columns (Normal, GT-fed, GT), N rows (selected frames)
    frames_to_plot = [f for f in FRAME_INDICES if f < len(samples)]
    rows = len(frames_to_plot)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 4 * rows))
    if rows == 1: axes = axes[None, :]

    for ri, fidx in enumerate(frames_to_plot):
        tok = samples[fidx]
        info = token_to_info[tok]
        ego_xy = np.array(info['ego2global_translation'][:2])
        sample = nusc.get('sample', tok)

        # GT boxes
        gt_boxes = []  # list of (cx, cy, w, l, yaw, inst_token)
        for ann_tok in sample['anns']:
            ann = nusc.get('sample_annotation', ann_tok)
            cat = ann['category_name']
            if not any(n in cat for n in ['pedestrian','car','truck','bus','trailer','motorcycle','bicycle']):
                continue
            cx, cy = ann['translation'][:2]
            w, l = ann['size'][:2]
            q = Q(ann['rotation'])
            yaw = np.arctan2(q.rotation_matrix[1,0], q.rotation_matrix[0,0])
            gt_boxes.append((cx, cy, w, l, yaw, ann['instance_token']))

        # Tracker boxes (convert from LiDAR-frame to global)
        def convert_tracker(result):
            out = []
            for r in result:
                pos, size, rot, _ = lidar_to_global(r['bbox'].cpu().numpy().astype(float), info)
                cx, cy = pos[:2]
                w, l = size[0], size[1]
                yaw = np.arctan2(rot.rotation_matrix[1,0], rot.rotation_matrix[0,0])
                out.append((cx, cy, w, l, yaw, r['tracking_id']))
            return out

        normal_boxes = convert_tracker(normal_out[fidx])
        gtfed_boxes = convert_tracker(gtfed_out[fidx])

        # Centering: show 60m around ego
        x0, x1 = ego_xy[0] - 60, ego_xy[0] + 60
        y0, y1 = ego_xy[1] - 60, ego_xy[1] + 60

        titles = [f'Normal (f{fidx})', f'GT-fed (f{fidx})', f'GT (f{fidx})']
        box_sets = [normal_boxes, gtfed_boxes, gt_boxes]
        for ci, (title, boxes) in enumerate(zip(titles, box_sets)):
            ax = axes[ri, ci]
            for (cx, cy, w, l, yaw, key) in boxes:
                draw_box(ax, cx, cy, w, l, yaw, hash_color(key), lw=1.8)
            ax.plot(ego_xy[0], ego_xy[1], 'k*', markersize=10, zorder=5)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
            ax.set_aspect('equal')
            ax.set_title(f'{title} n={len(boxes)}', fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.3)
            ax.tick_params(labelsize=6)

    plt.suptitle(f'{SCENE}: color = persistent ID (tracking_id or instance_token)',
                 y=1.001, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT, dpi=100, bbox_inches='tight')
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
