"""Stage 2: Run MOTIP tracker on pre-extracted features + nuScenes eval.

Expects features extracted via the STANDARD mmdet pipeline (NOT
extract_track_feats.py — that script's output format is incompatible).

Full eval pipeline (Stage 1 + Stage 2):

    # Stage 1: extract raw 428 features per sample using dist_test.sh
    CUDA_VISIBLE_DEVICES=<GPU> bash tools/dist_test.sh \
        <CONFIG> <CKPT> 1 --out <WORK_DIR>/track_feats.pkl

    # Stage 2: run tracker + nuScenes eval
    CUDA_VISIBLE_DEVICES=<GPU> python tools/eval_tracking.py \
        --config <CONFIG> --checkpoint <CKPT> \
        --feats <WORK_DIR>/track_feats.pkl \
        --det-thresh 0.25 --new-thresh 0.40 --id-thresh 0.10 \
        --out <WORK_DIR>/tracking_results.json

Best thresholds (from sweep_thresholds tuning): det=0.25 new=0.40 id=0.10.
"""
import os, sys, json, argparse, pickle, logging
from collections import defaultdict

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
TRACKING_NAMES = {'car', 'truck', 'bus', 'trailer', 'pedestrian', 'motorcycle', 'bicycle'}
ATTR_MAP = {
    'car': 'vehicle.moving', 'truck': 'vehicle.moving',
    'construction_vehicle': 'vehicle.moving', 'bus': 'vehicle.moving',
    'trailer': 'vehicle.moving', 'barrier': '',
    'motorcycle': 'cycle.with_rider', 'bicycle': 'cycle.with_rider',
    'pedestrian': 'pedestrian.moving', 'traffic_cone': '',
}


class MOTIPTracker:
    """Online tracker using MOTIP ID prediction module."""

    def __init__(self, model, motip_cfg):
        self.model = model
        self.K = motip_cfg['num_ids']
        self.det_thresh = motip_cfg.get('det_thresh', 0.3)
        self.new_thresh = motip_cfg.get('new_thresh', 0.6)
        self.id_thresh = motip_cfg.get('id_thresh', 0.2)
        self.max_age = motip_cfg.get('max_age', 5)
        self.context_len = motip_cfg.get('context_len', 5)
        self.reset()

    def reset(self):
        self.tracks = {}
        self.next_global_id = 0
        self.slot_to_global = {}
        self.global_to_slot = {}
        self.history = []  # list of {'feat', 'bbox', 'id_emb', 'ego_pose'}

    def _allocate_slot(self, global_id):
        for s in range(self.K):
            if s not in self.slot_to_global:
                self.slot_to_global[s] = global_id
                self.global_to_slot[global_id] = s
                return s
        oldest_age, oldest_slot = -1, -1
        for s, gid in self.slot_to_global.items():
            if gid in self.tracks and self.tracks[gid]['age'] > oldest_age:
                oldest_age = self.tracks[gid]['age']
                oldest_slot = s
        if oldest_slot >= 0:
            old_gid = self.slot_to_global[oldest_slot]
            del self.slot_to_global[oldest_slot]
            del self.global_to_slot[old_gid]
            self.slot_to_global[oldest_slot] = global_id
            self.global_to_slot[global_id] = oldest_slot
            return oldest_slot
        return 0

    def _free_slot(self, global_id):
        if global_id in self.global_to_slot:
            slot = self.global_to_slot.pop(global_id)
            self.slot_to_global.pop(slot, None)

    @torch.no_grad()
    def track_frame(self, sample_data, ego_pose):
        """Process one frame from pre-extracted features.

        sample_data: dict with scores, labels, bbox_decoded, bbox_raw, query_feat
        ego_pose: [4, 4] lidar2global tensor
        """
        m = self.model
        device = next(m.parameters()).device

        scores = sample_data['scores'].to(device)
        labels = sample_data['labels'].to(device)
        bbox_decoded = sample_data['bbox_decoded'].to(device)
        bbox_raw = sample_data['bbox_raw'].to(device)
        query_feat = sample_data['query_feat'].to(device)
        ego_pose = ego_pose.to(device)

        # Filter by det_thresh
        mask = scores > self.det_thresh
        scores_f = scores[mask]
        labels_f = labels[mask]
        bbox_decoded_f = bbox_decoded[mask]
        bbox_raw_f = bbox_raw[mask]
        det_feat = query_feat[mask]

        N_det = len(scores_f)
        if N_det == 0:
            dead_ids = []
            for gid, t in self.tracks.items():
                t['age'] += 1
                if t['age'] > self.max_age:
                    dead_ids.append(gid)
            for gid in dead_ids:
                self._free_slot(gid)
                del self.tracks[gid]
            return []

        # PE_3D
        pe_input = m._bbox_to_pe_input(bbox_raw_f)
        pe = m.pe_3d(pe_input[:, :7], pe_input[:, 7:9])

        has_history = len(self.history) > 0

        if not has_history:
            results = []
            for i in range(N_det):
                gid = self.next_global_id
                self.next_global_id += 1
                slot = self._allocate_slot(gid)
                self.tracks[gid] = {
                    'id_slot': slot, 'age': 0,
                    'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(),
                }
                results.append({
                    'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(), 'tracking_id': gid,
                })

            slot_ids = torch.tensor(
                [self.tracks[r['tracking_id']]['id_slot'] for r in results],
                device=device, dtype=torch.long)
            id_emb = m.id_dict.get_id_embedding(slot_ids)
            self.history.append({
                'feat': det_feat.clone(), 'bbox': bbox_raw_f.clone(),
                'id_emb': id_emb.clone(), 'ego_pose': ego_pose.clone(),
            })
            if len(self.history) > self.context_len:
                self.history.pop(0)
            return results

        # Build context with coordinate transform
        ctx_f, ctx_p, ctx_i = [], [], []
        for h in self.history:
            ctx_bbox_cur = m._transform_bbox_to_current(
                h['bbox'], h['ego_pose'], ego_pose)
            ctx_pe_in = m._bbox_to_pe_input(ctx_bbox_cur)
            ctx_pe = m.pe_3d(ctx_pe_in[:, :7], ctx_pe_in[:, 7:9])
            ctx_f.append(h['feat'])
            ctx_p.append(ctx_pe)
            ctx_i.append(h['id_emb'])

        context = m.tracklet_former.form_tracklet(
            torch.cat(ctx_f), torch.cat(ctx_p), torch.cat(ctx_i)
        ).unsqueeze(0)

        spec = m.id_dict.get_special_token(N_det)
        queries = m.tracklet_former.form_tracklet(
            det_feat, pe, spec
        ).unsqueeze(0)

        id_logits = m.id_decoder(queries, context)
        id_probs = torch.softmax(id_logits.squeeze(0), dim=-1)

        # Greedy assignment
        results = []
        assigned_slots = set()
        sort_idx = scores_f.argsort(descending=True)

        for idx in sort_idx:
            i = idx.item()
            probs = id_probs[i]
            existing_probs = probs[:self.K].clone()
            for s in assigned_slots:
                existing_probs[s] = -1

            best_slot = existing_probs.argmax().item()
            best_prob = existing_probs[best_slot].item()
            newborn_prob = probs[self.K].item()
            slot_has_track = best_slot in self.slot_to_global

            if slot_has_track and best_prob > newborn_prob and best_prob > self.id_thresh:
                gid = self.slot_to_global[best_slot]
                assigned_slots.add(best_slot)
            elif scores_f[i].item() > self.new_thresh:
                gid = self.next_global_id
                self.next_global_id += 1
                best_slot = self._allocate_slot(gid)
                assigned_slots.add(best_slot)
            else:
                continue

            self.tracks[gid] = {
                'id_slot': best_slot, 'age': 0,
                'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                'label': labels_f[i].item(),
            }
            results.append({
                'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                'label': labels_f[i].item(), 'tracking_id': gid,
                '_det_idx': i,
            })

        matched_gids = {r['tracking_id'] for r in results}
        dead_ids = []
        for gid, t in self.tracks.items():
            if gid not in matched_gids:
                t['age'] += 1
                if t['age'] > self.max_age:
                    dead_ids.append(gid)
        for gid in dead_ids:
            self._free_slot(gid)
            del self.tracks[gid]

        if results:
            result_indices = [r['_det_idx'] for r in results]
            hist_feat = det_feat[result_indices]
            hist_bbox = bbox_raw_f[result_indices]
            slot_ids = torch.tensor(
                [self.tracks[r['tracking_id']]['id_slot'] for r in results],
                device=device, dtype=torch.long)
            id_emb = m.id_dict.get_id_embedding(slot_ids)
            self.history.append({
                'feat': hist_feat.clone(), 'bbox': hist_bbox.clone(),
                'id_emb': id_emb.clone(), 'ego_pose': ego_pose.clone(),
            })
        if len(self.history) > self.context_len:
            self.history.pop(0)

        return results


def lidar_to_global(box_np, info):
    """Convert denormalize_bbox output (gravity center) to nuScenes global frame.

    Ported directly from SparseBEV/val_tracking.py:332-366 (known working).
    box_np: [cx, cy, cz, dx, dy, h, rot, vx, vy] — gravity center from denormalize_bbox.
    """
    from pyquaternion import Quaternion

    cx, cy, cz, dx, dy, h, rot, vx, vy = box_np[:9]
    # nuScenes size: [width, length, height] = [dy, dx, h]
    w, l = float(dy), float(dx)

    lidar2ego_rot = Quaternion(info['lidar2ego_rotation'])
    lidar2ego_trans = np.array(info['lidar2ego_translation'])
    ego2global_rot = Quaternion(info['ego2global_rotation'])
    ego2global_trans = np.array(info['ego2global_translation'])

    pos = ego2global_rot.rotate(
        lidar2ego_rot.rotate(np.array([cx, cy, cz])) + lidar2ego_trans
    ) + ego2global_trans
    rot_quat = ego2global_rot * lidar2ego_rot * Quaternion(axis=[0, 0, 1], angle=float(rot))
    vel = ego2global_rot.rotate(lidar2ego_rot.rotate(np.array([vx, vy, 0.0])))

    return pos.tolist(), [w, l, float(h)], rot_quat, vel[:2].tolist()


def build_tracking_submission(all_results, data_infos, token_to_info):
    """Build nuScenes tracking submission using direct coordinate transform."""
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
            if np.sqrt(vel[0]**2 + vel[1]**2) <= 0.2:
                if class_name == 'pedestrian':
                    attr = 'pedestrian.standing'
                elif class_name == 'bus':
                    attr = 'vehicle.stopped'

            submission_results[sample_token].append({
                'sample_token': sample_token,
                'translation': pos,
                'size': size,
                'rotation': rot_quat.elements.tolist(),
                'velocity': vel,
                'tracking_id': str(det['tracking_id']),
                'tracking_name': class_name,
                'tracking_score': det['score'],
                'attribute_name': attr,
            })

    return submission_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--feats', required=True, help='Path to track_feats.pkl')
    parser.add_argument('--out', default='tracking_results.json')
    parser.add_argument('--det-thresh', type=float, default=0.3)
    parser.add_argument('--new-thresh', type=float, default=0.6)
    parser.add_argument('--id-thresh', type=float, default=0.2)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        _module_dir = os.path.dirname(cfg.plugin_dir).split('/')
        importlib.import_module('.'.join(_module_dir))

    # Load model (MOTIP modules only needed, but load full for simplicity)
    model = build_model(cfg.model)
    wrap_fp16_model(model)
    model.cuda().eval()
    log.info('Loading checkpoint: %s', args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cuda', strict=False)

    # Load pre-extracted features (list from dist_test, index-aligned with val infos)
    log.info('Loading features: %s', args.feats)
    with open(args.feats, 'rb') as f:
        raw_outputs = pickle.load(f)
    log.info('Loaded %d samples', len(raw_outputs))

    # Load infos — use dataset.data_infos (matches DataLoader order)
    # NOT the raw pkl infos (different ordering!)
    val_cfg = cfg.data.val.copy()
    val_cfg['test_mode'] = True
    val_dataset = build_dataset(val_cfg)
    data_infos = val_dataset.data_infos

    with open(cfg.data.val.ann_file, 'rb') as f:
        val_data = pickle.load(f)
    token_to_info = {info['token']: info for info in val_data['infos']}

    # Convert list outputs → token-keyed dict with tracking-ready format
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    all_feats = {}
    for idx, out in enumerate(raw_outputs):
        info = data_infos[idx]
        token = info['token']
        pts = out['pts_bbox']

        query_feat = pts['query_feat']    # [428, 256]
        bbox_raw = pts['bbox_raw']        # [428, 10]
        cls_scores = pts['cls_scores']    # [428, 10]

        # Top-k: same logic as NMSFreeCoder.decode
        cls_sig = cls_scores.sigmoid()
        num_classes = cls_scores.shape[1]
        scores_flat, indexs = cls_sig.view(-1).topk(300)
        labels = indexs % num_classes
        query_indices = torch.div(indexs, num_classes, rounding_mode='trunc')

        decoded_bbox = denormalize_bbox(bbox_raw[query_indices], pc_range)

        # Build ego_pose from info
        from pyquaternion import Quaternion as Q
        l2e_r = Q(info['lidar2ego_rotation']).rotation_matrix
        l2e_t = np.array(info['lidar2ego_translation'])
        e2g_r = Q(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'])
        l2g = np.eye(4)
        l2g[:3, :3] = e2g_r @ l2e_r
        l2g[:3, 3] = e2g_r @ l2e_t + e2g_t

        all_feats[token] = {
            'scores': scores_flat,                         # [300]
            'labels': labels,                              # [300]
            'bbox_decoded': decoded_bbox,                  # [300, 9]
            'bbox_raw': bbox_raw[query_indices],           # [300, 10]
            'query_feat': query_feat[query_indices],       # [300, 256]
            'ego_pose': torch.from_numpy(l2g).float(),
        }
    # Keep raw_outputs for build_tracking_submission (coordinate transform)

    # Group by scene
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=cfg.data.val.data_root, verbose=False)
    val_scene_names = set(VAL_SCENES)
    scene_samples = defaultdict(list)
    for info in data_infos:
        sample = nusc.get('sample', info['token'])
        scene = nusc.get('scene', sample['scene_token'])
        if scene['name'] in val_scene_names:
            scene_samples[sample['scene_token']].append(
                (info['timestamp'], info['token']))
    for st in scene_samples:
        scene_samples[st].sort()

    # MOTIP tracker
    motip_cfg = dict(cfg.model.motip_cfg)
    motip_cfg['det_thresh'] = args.det_thresh
    motip_cfg['new_thresh'] = args.new_thresh
    motip_cfg['id_thresh'] = args.id_thresh
    tracker = MOTIPTracker(model, motip_cfg)

    log.info('Running tracker: %d scenes, %d samples',
             len(scene_samples), sum(len(v) for v in scene_samples.values()))

    all_results = {}
    for si, (scene_token, samples) in enumerate(scene_samples.items()):
        scene = nusc.get('scene', scene_token)
        if (si + 1) % 20 == 0 or si == 0:
            log.info('[%d/%d] %s (%d frames)',
                     si+1, len(scene_samples), scene['name'], len(samples))
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

    # Build submission using mmdet3d coordinate transforms (same path as det eval)
    log.info('Building submission...')
    submission_results = build_tracking_submission(all_results, data_infos, token_to_info)

    # Ensure all val samples have entries
    for scene_token, samples in scene_samples.items():
        for _, sample_token in samples:
            if sample_token not in submission_results:
                submission_results[sample_token] = []

    submission = {
        'meta': {'use_camera': True, 'use_lidar': False,
                 'use_radar': False, 'use_map': False, 'use_external': False},
        'results': dict(submission_results),
    }

    out_path = os.path.join(os.path.dirname(args.checkpoint), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(submission, f)
    log.info('Saved: %s', out_path)

    # nuScenes eval
    log.info('Running TrackingEval...')
    eval_cfg = config_factory('tracking_nips_2019')
    tracking_eval = TrackingEval(
        config=eval_cfg,
        result_path=out_path,
        eval_set='val',
        output_dir=os.path.dirname(out_path),
        nusc_version='v1.0-trainval',
        nusc_dataroot=cfg.data.val.data_root,
        verbose=True,
    )
    result = tracking_eval.main()

    summary = result if isinstance(result, dict) else result.serialize()
    log.info('=== Tracking Results ===')
    label_metrics = summary.get('label_metrics', {})
    for mn in ['amota', 'amotp', 'recall', 'mota', 'motp', 'ids', 'frag']:
        if mn in label_metrics:
            vals = label_metrics[mn]
            log.info('  %s: %.4f', mn, float(np.nanmean(list(vals.values()))))
    log.info('Done.')


if __name__ == '__main__':
    main()
