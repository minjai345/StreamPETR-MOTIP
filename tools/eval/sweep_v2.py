"""Threshold sweep for v2 fresh iter_28128, including max_age.

Loads pre-extracted features once, varies (det/new/id thresholds + max_age),
runs tracker per scene, computes nuScenes TrackingEval per config.
"""
import os, sys, json, pickle, logging, time
from collections import defaultdict
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

from tools.eval_tracking import MOTIPTracker, lidar_to_global, CLASS_NAMES, TRACKING_NAMES, ATTR_MAP

CFG = "projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e_phase2.py"
CKPT = "work_dirs/motip_e2e_v2_fresh/iter_28128.pth"
FEATS = "work_dirs/motip_e2e_v2_fresh/track_feats_iter28128.pkl"

cfg = Config.fromfile(CFG)
import importlib; importlib.import_module("projects.mmdet3d_plugin")

model = build_model(cfg.model)
wrap_fp16_model(model)
model.cuda().eval()
load_checkpoint(model, CKPT, map_location="cuda", strict=False)

with open(FEATS, "rb") as f:
    raw_outputs = pickle.load(f)

val_cfg = cfg.data.val.copy(); val_cfg["test_mode"] = True
dataset = build_dataset(val_cfg)
data_infos = dataset.data_infos
with open(cfg.data.val.ann_file, "rb") as f:
    val_data = pickle.load(f)
token_to_info = {info["token"]: info for info in val_data["infos"]}

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
pc_range = [-51.2,-51.2,-5,51.2,51.2,3]
all_feats = {}
for idx, out in enumerate(raw_outputs):
    info = data_infos[idx]; token = info["token"]
    pts = out["pts_bbox"]
    cls_sig = pts["cls_scores"].sigmoid()
    scores_flat, indexs = cls_sig.view(-1).topk(300)
    labels = indexs % 10
    qi = torch.div(indexs, 10, rounding_mode="trunc")
    decoded = denormalize_bbox(pts["bbox_raw"][qi], pc_range)
    from pyquaternion import Quaternion as Q
    l2e_r = Q(info['lidar2ego_rotation']).rotation_matrix
    l2e_t = np.array(info['lidar2ego_translation'])
    e2g_r = Q(info['ego2global_rotation']).rotation_matrix
    e2g_t = np.array(info['ego2global_translation'])
    l2g = np.eye(4); l2g[:3,:3] = e2g_r @ l2e_r; l2g[:3,3] = e2g_r @ l2e_t + e2g_t
    all_feats[token] = {
        'scores': scores_flat, 'labels': labels,
        'bbox_decoded': decoded, 'bbox_raw': pts["bbox_raw"][qi],
        'query_feat': pts["query_feat"][qi],
        'ego_pose': torch.from_numpy(l2g).float(),
    }
del raw_outputs

nusc = NuScenes(version="v1.0-trainval", dataroot=cfg.data.val.data_root, verbose=False)
val_scene_names = set(VAL_SCENES)
scene_samples = defaultdict(list)
for info in val_data["infos"]:
    sample = nusc.get("sample", info["token"])
    scene = nusc.get("scene", sample["scene_token"])
    if scene["name"] in val_scene_names:
        scene_samples[sample["scene_token"]].append((info["timestamp"], info["token"]))
for st in scene_samples: scene_samples[st].sort()

log.info("Data loaded. Starting sweep.")

# Sweep configs: (det_thresh, new_thresh, id_thresh, max_age)
configs = [
    (0.25, 0.40, 0.10, 5),    # baseline
    (0.20, 0.40, 0.10, 5),
    (0.15, 0.40, 0.10, 5),
    (0.20, 0.30, 0.10, 5),
    (0.15, 0.30, 0.10, 5),
    (0.20, 0.30, 0.05, 5),
    (0.20, 0.30, 0.10, 10),
    (0.20, 0.30, 0.10, 15),
    (0.15, 0.30, 0.10, 10),
    (0.20, 0.40, 0.10, 10),
    (0.20, 0.40, 0.10, 15),
    (0.25, 0.40, 0.10, 10),
]

results = []
for det_t, new_t, id_t, max_age in configs:
    t0 = time.time()
    mc = dict(cfg.model.motip_cfg)
    mc['det_thresh'] = det_t; mc['new_thresh'] = new_t
    mc['id_thresh'] = id_t; mc['max_age'] = max_age
    tracker = MOTIPTracker(model, mc)

    all_results = {}
    for scene_token, samples in scene_samples.items():
        tracker.reset()
        for _, tok in samples:
            if tok not in all_feats: continue
            fd = all_feats[tok]; ego = fd['ego_pose']
            if ego is None: continue
            all_results[tok] = tracker.track_frame(fd, ego)

    # Build submission
    submission = defaultdict(list)
    for tok, fr in all_results.items():
        if not fr: continue
        info = token_to_info[tok]
        for det in fr:
            bbox = det['bbox'].cpu().numpy().astype(float)
            cn = CLASS_NAMES[det['label']]
            if cn not in TRACKING_NAMES: continue
            pos, size, rot, vel = lidar_to_global(bbox, info)
            attr = ATTR_MAP.get(cn, '')
            if np.sqrt(vel[0]**2+vel[1]**2) <= 0.2:
                if cn == 'pedestrian': attr = 'pedestrian.standing'
                elif cn == 'bus': attr = 'vehicle.stopped'
            submission[tok].append({
                'sample_token': tok, 'translation': pos, 'size': size,
                'rotation': rot.elements.tolist(), 'velocity': vel,
                'tracking_id': str(det['tracking_id']), 'tracking_name': cn,
                'tracking_score': det['score'], 'attribute_name': attr,
            })
    for st, samples in scene_samples.items():
        for _, tok in samples:
            if tok not in submission: submission[tok] = []

    out_path = f"/tmp/sweep_v2_d{det_t}_n{new_t}_i{id_t}_a{max_age}.json"
    with open(out_path, "w") as f:
        json.dump({"meta":{"use_camera":True,"use_lidar":False,"use_radar":False,
                  "use_map":False,"use_external":False}, "results":dict(submission)}, f)

    eval_cfg = config_factory("tracking_nips_2019")
    te = TrackingEval(config=eval_cfg, result_path=out_path, eval_set="val",
                      output_dir="/tmp/sweep_v2_out/", nusc_version="v1.0-trainval",
                      nusc_dataroot=cfg.data.val.data_root, verbose=False)
    res = te.main()
    summ = res if isinstance(res, dict) else res.serialize()
    lm = summ.get("label_metrics", {})
    amota = np.nanmean(list(lm["amota"].values())) if "amota" in lm else 0
    mota = np.nanmean(list(lm["mota"].values())) if "mota" in lm else 0
    ids = int(np.nansum(list(lm["ids"].values()))) if "ids" in lm else 0
    tp = int(np.nansum(list(lm["tp"].values()))) if "tp" in lm else 0
    fp = int(np.nansum(list(lm["fp"].values()))) if "fp" in lm else 0
    el = time.time() - t0
    log.info(f"d={det_t} n={new_t} i={id_t} age={max_age} → AMOTA={amota:.4f} MOTA={mota:.4f} IDS={ids} TP={tp} FP={fp} ({el:.0f}s)")
    results.append((det_t, new_t, id_t, max_age, amota, mota, ids, tp, fp))

log.info("\n=== SWEEP RESULTS (sorted by AMOTA desc) ===")
log.info(f"{'det':>5} {'new':>5} {'id':>5} {'age':>4} {'AMOTA':>8} {'MOTA':>8} {'IDS':>6} {'TP':>6} {'FP':>6}")
for d, n, i, a, am, mo, ids, tp, fp in sorted(results, key=lambda x: -x[4]):
    log.info(f"{d:5.2f} {n:5.2f} {i:5.2f} {a:>4d} {am:8.4f} {mo:8.4f} {ids:>6d} {tp:>6d} {fp:>6d}")
