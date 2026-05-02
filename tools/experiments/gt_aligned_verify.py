"""Verify GTAlignedTracker: slot_to_global aligned with GT mapping,
history id_emb uses GT slots for matched detections.

Runs scene-0105 and prints:
  - matching rate per frame
  - slot_to_global contents (should reflect GT gids for registered instances)
  - id_emb shape / consistency between frames
  - sample assignments (decoder prediction vs GT slot)
"""
import os, sys, pickle
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from pyquaternion import Quaternion as Q

from tools.experiments.gt_aligned_eval import GTAlignedTracker
from tools.eval_tracking import lidar_to_global

CFG = "projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e_phase2.py"
CKPT = "work_dirs/motip_e2e_v2_fresh/iter_28128.pth"
FEATS = "work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl"
SCENE = "scene-0105"

cfg = Config.fromfile(CFG)
import importlib; importlib.import_module('projects.mmdet3d_plugin')
model = build_model(cfg.model); wrap_fp16_model(model)
model.cuda().eval()
load_checkpoint(model, CKPT, map_location='cuda', strict=False)

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
tracker = GTAlignedTracker(model, mc); tracker.reset()

print(f'Scene: {SCENE}, frames: {len(samples)}')
print(f'{"frame":>6} {"#pred":>6} {"matched":>8} {"#GT_regd":>10} {"#slots_used":>12}')

for fidx, tok in enumerate(samples):
    fd = all_feats[tok]
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
    tracker.set_frame_gt(gt_inst, gt_centers)
    tracker.set_frame_info(info)
    res = tracker.track_frame(fd, fd['ego_pose'])
    n_matched_in_results = sum(1 for r in res
                                if r['tracking_id'] in tracker.inst_to_gid.values())
    n_gt_registered = len(tracker.inst_to_gid)
    n_slots_used = len(tracker.slot_to_global)
    if fidx < 10 or fidx in [20, 30, 38]:
        print(f'{fidx:>6} {len(res):>6} {n_matched_in_results:>8} {n_gt_registered:>10} {n_slots_used:>12}')

# Verify invariants
print()
print('=== INVARIANTS ===')
# Each registered GT instance should have exactly one slot, and slot_to_global[slot] should be the gid
bad = 0
for inst, gid in tracker.inst_to_gid.items():
    slot = tracker.inst_to_slot.get(inst)
    if slot is None:
        bad += 1; continue
    if tracker.slot_to_global.get(slot) != gid:
        bad += 1; print(f'Mismatch: inst={inst[:12]} gid={gid} slot={slot} slot_to_global={tracker.slot_to_global.get(slot)}')
print(f'inst→gid consistency: {len(tracker.inst_to_gid) - bad}/{len(tracker.inst_to_gid)} OK')

# GT slots should have persistent gids across history
print(f'Total GT instances registered: {len(tracker.inst_to_gid)}')
print(f'Total slots used: {len(tracker.slot_to_global)}')
print(f'History length: {len(tracker.history)} (max context_len = {tracker.context_len})')

# Show some example GT instance → slot mapping
print()
print('Sample registered GT instances:')
for i, (inst, gid) in enumerate(list(tracker.inst_to_gid.items())[:5]):
    slot = tracker.inst_to_slot[inst]
    print(f'  inst={inst[:16]}... gid={gid} slot={slot} '
          f'slot_to_global[{slot}]={tracker.slot_to_global.get(slot)}')
