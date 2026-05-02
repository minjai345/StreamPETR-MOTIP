"""Verify oracle association tracker output:
  - Matched detections get 'gt_<instance_token>' tracking_id
  - Same GT instance → same tracking_id across frames (persistence)
  - Unmatched detections keep numeric predicted id
  - Match count reasonable (should be close to #predictions if detector decent)

Runs on one scene, prints diagnostics.
"""
import os, sys, pickle
from collections import defaultdict, Counter
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from pyquaternion import Quaternion as Q

from tools.eval_tracking import MOTIPTracker, lidar_to_global

CFG = "projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e_phase2.py"
CKPT = "work_dirs/motip_e2e_v2_fresh/iter_28128.pth"
FEATS = "work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl"
SCENE = "scene-0105"
MATCH_DIST = 2.0

cfg = Config.fromfile(CFG)
import importlib; importlib.import_module('projects.mmdet3d_plugin')
model = build_model(cfg.model); wrap_fp16_model(model)
model.cuda().eval()
load_checkpoint(model, CKPT, map_location='cuda', strict=False)

# Feats
raw = pickle.load(open(FEATS, 'rb'))
val_cfg = cfg.data.val.copy(); val_cfg['test_mode'] = True
dataset = build_dataset(val_cfg)
data_infos = dataset.data_infos
val_data = pickle.load(open(cfg.data.val.ann_file, 'rb'))
token_to_info = {info['token']: info for info in val_data['infos']}

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
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

nusc = NuScenes(version='v1.0-trainval', dataroot=cfg.data.val.data_root, verbose=False)
scene = next(s for s in nusc.scene if s['name'] == SCENE)
samples = []; tok = scene['first_sample_token']
while tok: samples.append(tok); tok = nusc.get('sample', tok)['next']

mc = dict(cfg.model.motip_cfg)
mc.update({'det_thresh':0.25,'new_thresh':0.40,'id_thresh':0.10,'max_age':5})
tracker = MOTIPTracker(model, mc); tracker.reset()

# Replicate oracle association loop from eval script
frame_results_log = []
for fidx, tok in enumerate(samples):
    if tok not in all_feats: continue
    fd = all_feats[tok]
    results = tracker.track_frame(fd, fd['ego_pose'])
    sample = nusc.get('sample', tok)
    info = token_to_info[tok]

    gt_inst, gt_centers = [], []
    for ann_tok in sample['anns']:
        ann = nusc.get('sample_annotation', ann_tok)
        cat = ann['category_name']
        if not any(n in cat for n in ['pedestrian','car','truck','bus',
                                      'trailer','motorcycle','bicycle']):
            continue
        gt_inst.append(ann['instance_token'])
        gt_centers.append(ann['translation'][:2])

    pred_centers = []
    for r in results:
        pos, _, _, _ = lidar_to_global(r['bbox'].cpu().numpy().astype(float), info)
        pred_centers.append(pos[:2])
    pred_centers = np.array(pred_centers) if pred_centers else np.zeros((0,2))
    gt_centers_arr = np.array(gt_centers) if gt_centers else np.zeros((0,2))

    pred_to_gt = [-1] * len(results)
    if len(pred_centers) > 0 and len(gt_centers_arr) > 0:
        dists = np.linalg.norm(pred_centers[:, None, :] - gt_centers_arr[None, :, :], axis=-1)
        used = set()
        for pi in np.argsort(dists.min(axis=1)):
            if dists[pi].min() > MATCH_DIST: continue
            for gi in np.argsort(dists[pi]):
                if gi in used or dists[pi, gi] > MATCH_DIST: continue
                pred_to_gt[pi] = gi
                used.add(gi)
                break

    frame = []
    for i, r in enumerate(results):
        orig_tid = r['tracking_id']
        gi = pred_to_gt[i]
        if gi >= 0:
            new_tid = 'gt_' + gt_inst[gi]
        else:
            new_tid = str(orig_tid)
        frame.append({'orig_tid': orig_tid, 'new_tid': new_tid,
                      'matched_gt_inst': gt_inst[gi] if gi >= 0 else None})
    frame_results_log.append(frame)

# Analysis
print(f'Scene: {SCENE}, frames: {len(frame_results_log)}')
total_pred = sum(len(f) for f in frame_results_log)
total_matched = sum(1 for f in frame_results_log for r in f if r['matched_gt_inst'])
print(f'Total predictions: {total_pred}')
print(f'Matched to GT: {total_matched} ({100*total_matched/total_pred:.1f}%)')

# Verify: matched ones have 'gt_' prefix
bad = sum(1 for f in frame_results_log for r in f
          if r['matched_gt_inst'] and not r['new_tid'].startswith('gt_'))
print(f'Matched without gt_ prefix: {bad} (should be 0)')

# Verify: same GT instance → same tracking_id
inst_to_tids = defaultdict(set)
for f in frame_results_log:
    for r in f:
        if r['matched_gt_inst']:
            inst_to_tids[r['matched_gt_inst']].add(r['new_tid'])
bad2 = sum(1 for tids in inst_to_tids.values() if len(tids) > 1)
print(f'GT instances with inconsistent tracking_id: {bad2} (should be 0)')

# Show a few examples
print('\nFirst 5 frames, first 3 preds each:')
for fi, frame in enumerate(frame_results_log[:5]):
    print(f'  Frame {fi}:')
    for r in frame[:3]:
        marker = '★MATCHED' if r['matched_gt_inst'] else ' '
        inst_short = r['matched_gt_inst'][:12] + '...' if r['matched_gt_inst'] else '-'
        print(f'    orig={r["orig_tid"]:<4} new={r["new_tid"][:20]:<20} {marker} inst={inst_short}')

# Sample a persistence example: pick most-tracked GT instance
if inst_to_tids:
    most_frequent = max(inst_to_tids, key=lambda k: sum(
        1 for f in frame_results_log for r in f if r['matched_gt_inst'] == k))
    n_appear = sum(1 for f in frame_results_log for r in f
                   if r['matched_gt_inst'] == most_frequent)
    print(f'\nMost-tracked GT instance: {most_frequent[:16]}... '
          f'appears in {n_appear} detections')
    print(f'  tracking_ids assigned: {inst_to_tids[most_frequent]}')
    print(f'  expected: 1 unique id  → actual: {len(inst_to_tids[most_frequent])}')
