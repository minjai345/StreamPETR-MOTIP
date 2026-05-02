"""Quick verification that GTFeedingTracker actually overrides id_emb.

Runs 1 scene with both GTFeedingTracker and normal MOTIPTracker, compares:
  - tracking_id outputs (can differ, expected)
  - history id_emb tensors (should differ when override happens)
  - override counter (should be > 0)
"""
import os, sys, pickle, logging
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES

from tools.eval_tracking import MOTIPTracker, lidar_to_global
from tools.experiments.gt_feeding_eval import GTFeedingTracker

CFG = "projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e_phase2.py"
CKPT = "work_dirs/motip_e2e_v2_fresh/iter_28128.pth"
FEATS = "work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl"
TARGET_SCENE = "scene-0105"  # worst scene

cfg = Config.fromfile(CFG)
import importlib; importlib.import_module('projects.mmdet3d_plugin')

model = build_model(cfg.model)
wrap_fp16_model(model)
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
    from pyquaternion import Quaternion as Q
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
target_scene = next(s for s in nusc.scene if s['name'] == TARGET_SCENE)

samples = []
tok = target_scene['first_sample_token']
while tok:
    samples.append(tok); tok = nusc.get('sample', tok)['next']

mc = dict(cfg.model.motip_cfg)
mc['det_thresh'] = 0.25; mc['new_thresh'] = 0.40
mc['id_thresh'] = 0.10; mc['max_age'] = 5

# === Run 1: Normal MOTIPTracker ===
torch.manual_seed(0)
normal = MOTIPTracker(model, mc)
normal.reset()
normal_results = []
normal_history_emb = []
for tok in samples:
    if tok not in all_feats: continue
    fd = all_feats[tok]; ego = fd['ego_pose']
    fr = normal.track_frame(fd, ego)
    normal_results.append([(r['tracking_id'], r['bbox'].cpu().numpy()[:3]) for r in fr])
    if len(normal.history) > 0:
        normal_history_emb.append(normal.history[-1]['id_emb'].detach().cpu().clone())

# === Run 2: GTFeedingTracker ===
torch.manual_seed(0)
gtfed = GTFeedingTracker(model, mc)
gtfed.reset(); gtfed.reset_scene_gt()
gtfed_results = []
gtfed_history_emb = []
override_counter = 0
for tok in samples:
    if tok not in all_feats: continue
    fd = all_feats[tok]; ego = fd['ego_pose']
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
    fr = gtfed.track_frame(fd, ego)
    gtfed_results.append([(r['tracking_id'], r['bbox'].cpu().numpy()[:3]) for r in fr])
    if len(gtfed.history) > 0:
        gtfed_history_emb.append(gtfed.history[-1]['id_emb'].detach().cpu().clone())

# === Compare ===
log.info(f'Scene: {TARGET_SCENE}, frames: {len(samples)}')
log.info(f'Normal tracker: unique tracking_ids = {len(set(tid for fr in normal_results for tid,_ in fr))}')
log.info(f'GT-fed tracker: unique tracking_ids = {len(set(tid for fr in gtfed_results for tid,_ in fr))}')
log.info(f'GT instance map size: {len(gtfed.gt_inst_to_slot)}')

# Check id_emb differences
n_diff = 0
n_same = 0
for i, (ne, ge) in enumerate(zip(normal_history_emb, gtfed_history_emb)):
    if ne.shape != ge.shape:
        log.info(f'  Frame {i}: shape differs {ne.shape} vs {ge.shape}')
        continue
    if torch.allclose(ne, ge, atol=1e-6):
        n_same += 1
    else:
        n_diff += 1
        max_diff = (ne - ge).abs().max().item()
        if n_diff <= 3:
            log.info(f'  Frame {i}: id_emb differs, max|diff|={max_diff:.6f}')

log.info(f'History id_emb: {n_diff} frames differ, {n_same} same')

# Per-frame tracking_id comparison
log.info('')
log.info('Per-frame comparison:')
log.info(f'{"frame":>6} {"#normal":>8} {"#gtfed":>8} {"n_tids":>8} {"g_tids":>8}')
for i, (nr, gr) in enumerate(zip(normal_results, gtfed_results)):
    if i >= 10: break  # first 10 frames
    n_tids = {tid for tid, _ in nr}
    g_tids = {tid for tid, _ in gr}
    log.info(f'{i:>6} {len(nr):>8} {len(gr):>8} {len(n_tids):>8} {len(g_tids):>8}')

total_n = sum(len(fr) for fr in normal_results)
total_g = sum(len(fr) for fr in gtfed_results)
log.info(f'Total predictions: normal={total_n}, gtfed={total_g}')
