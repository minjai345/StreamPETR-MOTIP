"""Sweep tracker thresholds on pre-loaded features."""
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

# Import after path setup
from tools.eval_tracking import MOTIPTracker, lidar_to_global, CLASS_NAMES, TRACKING_NAMES, ATTR_MAP

cfg = Config.fromfile("projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e.py")
import importlib
importlib.import_module("projects.mmdet3d_plugin")

# Load model once
model = build_model(cfg.model)
wrap_fp16_model(model)
model.cuda().eval()
ckpt = "work_dirs/motip_phase1_v2_scenfix/iter_3516.pth"
load_checkpoint(model, ckpt, map_location="cuda", strict=False)

# Load features once
with open("work_dirs/motip_phase1_v1/track_feats.pkl", "rb") as f:
    raw_outputs = pickle.load(f)

# Build dataset for correct ordering
val_cfg = cfg.data.val.copy()
val_cfg["test_mode"] = True
dataset = build_dataset(val_cfg)
data_infos = dataset.data_infos

with open(cfg.data.val.ann_file, "rb") as f:
    val_data = pickle.load(f)
token_to_info = {info["token"]: info for info in val_data["infos"]}

# Convert features once
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
pc_range = [-51.2,-51.2,-5,51.2,51.2,3]
all_feats = {}
for idx, out in enumerate(raw_outputs):
    info = data_infos[idx]
    token = info["token"]
    pts = out["pts_bbox"]
    cls_sig = pts["cls_scores"].sigmoid()
    scores_flat, indexs = cls_sig.view(-1).topk(300)
    labels = indexs % 10
    query_indices = torch.div(indexs, 10, rounding_mode="trunc")
    decoded = denormalize_bbox(pts["bbox_raw"][query_indices], pc_range)
    from pyquaternion import Quaternion as Q
    l2e_r = Q(info['lidar2ego_rotation']).rotation_matrix
    l2e_t = np.array(info['lidar2ego_translation'])
    e2g_r = Q(info['ego2global_rotation']).rotation_matrix
    e2g_t = np.array(info['ego2global_translation'])
    l2g = np.eye(4)
    l2g[:3,:3] = e2g_r @ l2e_r
    l2g[:3,3] = e2g_r @ l2e_t + e2g_t
    all_feats[token] = {
        'scores': scores_flat, 'labels': labels,
        'bbox_decoded': decoded, 'bbox_raw': pts["bbox_raw"][query_indices],
        'query_feat': pts["query_feat"][query_indices],
        'ego_pose': torch.from_numpy(l2g).float(),
    }
del raw_outputs

# Group by scene
nusc = NuScenes(version="v1.0-trainval", dataroot=cfg.data.val.data_root, verbose=False)
val_scene_names = set(VAL_SCENES)
scene_samples = defaultdict(list)
for info in val_data["infos"]:
    sample = nusc.get("sample", info["token"])
    scene = nusc.get("scene", sample["scene_token"])
    if scene["name"] in val_scene_names:
        scene_samples[sample["scene_token"]].append((info["timestamp"], info["token"]))
for st in scene_samples:
    scene_samples[st].sort()

log.info("Data loaded. Starting sweep.")

# Sweep
configs = [
    (0.10, 0.20, 0.05), (0.10, 0.20, 0.10), (0.10, 0.30, 0.10),
    (0.15, 0.20, 0.10), (0.15, 0.30, 0.05), (0.15, 0.30, 0.10),
    (0.15, 0.30, 0.20), (0.15, 0.40, 0.10), (0.15, 0.50, 0.10),
    (0.20, 0.30, 0.10), (0.20, 0.40, 0.10), (0.20, 0.50, 0.10),
    (0.25, 0.30, 0.10), (0.25, 0.40, 0.10), (0.25, 0.50, 0.10),
]

results = []
for det_t, new_t, id_t in configs:
    t0 = time.time()
    motip_cfg = dict(cfg.model.motip_cfg)
    motip_cfg['det_thresh'] = det_t
    motip_cfg['new_thresh'] = new_t
    motip_cfg['id_thresh'] = id_t
    tracker = MOTIPTracker(model, motip_cfg)

    all_results = {}
    for si, (scene_token, samples) in enumerate(scene_samples.items()):
        tracker.reset()
        for _, sample_token in samples:
            if sample_token not in all_feats:
                continue
            feat_data = all_feats[sample_token]
            ego_pose = feat_data['ego_pose']
            if ego_pose is None:
                continue
            frame_results = tracker.track_frame(feat_data, ego_pose)
            all_results[sample_token] = frame_results

    # Build submission
    submission_results = defaultdict(list)
    for sample_token, frame_results in all_results.items():
        if not frame_results:
            continue
        info = token_to_info[sample_token]
        for det in frame_results:
            bbox = det['bbox'].cpu().numpy().astype(float)
            class_name = CLASS_NAMES[det['label']]
            if class_name not in TRACKING_NAMES:
                continue
            pos, size, rot_quat, vel = lidar_to_global(bbox, info)
            attr = ATTR_MAP.get(class_name, '')
            if np.sqrt(vel[0]**2+vel[1]**2) <= 0.2:
                if class_name == 'pedestrian': attr = 'pedestrian.standing'
                elif class_name == 'bus': attr = 'vehicle.stopped'
            submission_results[sample_token].append({
                'sample_token': sample_token, 'translation': pos, 'size': size,
                'rotation': rot_quat.elements.tolist(), 'velocity': vel,
                'tracking_id': str(det['tracking_id']), 'tracking_name': class_name,
                'tracking_score': det['score'], 'attribute_name': attr,
            })
    for scene_token, samples in scene_samples.items():
        for _, sample_token in samples:
            if sample_token not in submission_results:
                submission_results[sample_token] = []

    out_path = f"/tmp/sweep_det{det_t}_new{new_t}_id{id_t}.json"
    with open(out_path, "w") as f:
        json.dump({"meta": {"use_camera":True,"use_lidar":False,"use_radar":False,"use_map":False,"use_external":False}, "results": dict(submission_results)}, f)

    # Eval
    eval_cfg = config_factory("tracking_nips_2019")
    te = TrackingEval(config=eval_cfg, result_path=out_path, eval_set="val",
                      output_dir="/tmp/", nusc_version="v1.0-trainval",
                      nusc_dataroot=cfg.data.val.data_root, verbose=False)
    result = te.main()
    summary = result if isinstance(result, dict) else result.serialize()
    lm = summary.get("label_metrics", {})
    amota = np.nanmean(list(lm["amota"].values())) if "amota" in lm else 0
    mota = np.nanmean(list(lm["mota"].values())) if "mota" in lm else 0
    # IDS, TP, FP are counts — sum across classes, not average
    ids = int(np.nansum(list(lm["ids"].values()))) if "ids" in lm else 0
    tp = int(np.nansum(list(lm["tp"].values()))) if "tp" in lm else 0
    fp = int(np.nansum(list(lm["fp"].values()))) if "fp" in lm else 0

    elapsed = time.time() - t0
    log.info(f"det={det_t} new={new_t} id={id_t} → AMOTA={amota:.4f} MOTA={mota:.4f} IDS_total={ids} TP={tp} FP={fp} ({elapsed:.0f}s)")
    results.append((det_t, new_t, id_t, amota, mota, ids, tp, fp))

log.info("\n=== SWEEP RESULTS ===")
log.info(f"{'det':>5} {'new':>5} {'id':>5} {'AMOTA':>8} {'MOTA':>8} {'IDS_tot':>8} {'TP':>8} {'FP':>8}")
for det_t, new_t, id_t, amota, mota, ids, tp, fp in sorted(results, key=lambda x: -x[3]):
    log.info(f"{det_t:5.2f} {new_t:5.2f} {id_t:5.2f} {amota:8.4f} {mota:8.4f} {ids:8d} {tp:8d} {fp:8d}")
