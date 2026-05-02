"""Tracking evaluation for StreamPETR + MOTIP.

Runs per-scene sequential inference with MOTIP ID prediction,
produces nuScenes tracking submission JSON, and computes AMOTA metrics.

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/val_tracking.py \
        --config projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e.py \
        --checkpoint work_dirs/motip_phase1_v1/iter_14064.pth
"""

import os
import sys
import json
import argparse
import pickle
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory

# Add project root to path
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
        self.model = model.module if hasattr(model, 'module') else model
        self.cfg = motip_cfg
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
        self.history = []

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
    def track_frame(self, outs, img_metas, ego_pose):
        """Process one frame. ego_pose: [4,4] lidar2global for this frame."""
        device = outs['query_feat'].device
        m = self.model

        cls_scores = outs['all_cls_scores'][-1][0]   # [Q, num_cls]
        bbox_preds = outs['all_bbox_preds'][-1][0]    # [Q, 10]
        query_feat = outs['query_feat'][0]             # [Q, C]

        # Top-k decoding
        from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

        cls_sig = cls_scores.sigmoid()
        max_num = 300
        scores_flat, indexs = cls_sig.view(-1).topk(max_num)
        num_classes = cls_scores.shape[1]
        labels = indexs % num_classes
        query_indices = torch.div(indexs, num_classes, rounding_mode='trunc')

        raw_bbox = bbox_preds[query_indices]
        decoded_bbox = denormalize_bbox(raw_bbox, pc_range)

        mask = scores_flat > self.det_thresh
        scores_f = scores_flat[mask]
        labels_f = labels[mask]
        decoded_bbox_f = decoded_bbox[mask]
        query_indices_f = query_indices[mask]
        raw_bbox_f = bbox_preds[query_indices_f]

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

        # PE_3D from raw (normalized) bbox
        pe_input = m._bbox_to_pe_input(raw_bbox_f)
        pe = m.pe_3d(pe_input[:, :7], pe_input[:, 7:9])

        det_feat = query_feat[query_indices_f]

        has_history = len(self.history) > 0

        if not has_history:
            results = []
            for i in range(N_det):
                gid = self.next_global_id
                self.next_global_id += 1
                slot = self._allocate_slot(gid)
                self.tracks[gid] = {
                    'feat': det_feat[i], 'pe': pe[i], 'id_slot': slot,
                    'age': 0, 'bbox': decoded_bbox_f[i],
                    'score': scores_f[i].item(), 'label': labels_f[i].item(),
                }
                results.append({
                    'bbox': decoded_bbox_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(), 'tracking_id': gid,
                })

            slot_ids = torch.tensor(
                [self.tracks[r['tracking_id']]['id_slot'] for r in results],
                device=device, dtype=torch.long)
            id_emb = m.id_dict.get_id_embedding(slot_ids)
            # Store raw bbox + ego_pose for coordinate transform later
            self.history.append({
                'feat': det_feat.clone(), 'bbox': raw_bbox_f.clone(),
                'id_emb': id_emb.clone(), 'ego_pose': ego_pose.clone(),
            })
            if len(self.history) > self.context_len:
                self.history.pop(0)
            return results

        # Build context — transform history bbox to current frame coords
        ctx_f, ctx_p, ctx_i = [], [], []
        cur_l2g = ego_pose
        for h in self.history:
            ctx_bbox_cur = m._transform_bbox_to_current(
                h['bbox'], h['ego_pose'], cur_l2g)
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
                'feat': det_feat[i], 'pe': pe[i], 'id_slot': best_slot,
                'age': 0, 'bbox': decoded_bbox_f[i],
                'score': scores_f[i].item(), 'label': labels_f[i].item(),
            }
            results.append({
                'bbox': decoded_bbox_f[i], 'score': scores_f[i].item(),
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
            hist_bbox = raw_bbox_f[result_indices]
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
    """Convert decoded LiDAR-frame box [cx,cy,cz,w,l,h,rot,vx,vy] to global."""
    cx, cy, cz, w, l, h, rot = box_np[:7]
    vx, vy = (box_np[7], box_np[8]) if len(box_np) > 7 else (0, 0)

    lidar2ego_rot = Quaternion(info['lidar2ego_rotation'])
    lidar2ego_trans = np.array(info['lidar2ego_translation'])
    ego2global_rot = Quaternion(info['ego2global_rotation'])
    ego2global_trans = np.array(info['ego2global_translation'])

    pos = ego2global_rot.rotate(lidar2ego_rot.rotate(np.array([cx, cy, cz])) + lidar2ego_trans) + ego2global_trans
    rot_quat = ego2global_rot * lidar2ego_rot * Quaternion(axis=[0, 0, 1], angle=float(rot))
    vel = ego2global_rot.rotate(lidar2ego_rot.rotate(np.array([vx, vy, 0.0])))

    # nuScenes size: [width, length, height]
    return pos.tolist(), [float(l), float(w), float(h)], rot_quat, vel[:2].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='tracking_results.json')
    parser.add_argument('--det-thresh', type=float, default=0.3)
    parser.add_argument('--new-thresh', type=float, default=0.6)
    parser.add_argument('--id-thresh', type=float, default=0.2)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # Register plugin modules
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            _module_dir = os.path.dirname(cfg.plugin_dir).split('/')
            _module_path = '.'.join(_module_dir)
            importlib.import_module(_module_path)

    # Build model
    model = build_model(cfg.model)
    wrap_fp16_model(model)
    model.cuda()

    log.info('Loading checkpoint: %s', args.checkpoint)
    ckpt = load_checkpoint(model, args.checkpoint, map_location='cuda', strict=False)
    model.eval()
    model = MMDataParallel(model, device_ids=[0])
    inner = model.module

    # Build val dataset
    val_cfg = cfg.data.val.copy()
    val_cfg['test_mode'] = True
    val_dataset = build_dataset(val_cfg)

    # Load infos for coordinate transforms
    with open(cfg.data.val.ann_file, 'rb') as f:
        val_data = pickle.load(f)
    data_infos = val_data['infos']
    token_to_info = {info['token']: info for info in data_infos}
    token_to_dataset_idx = {}
    for idx in range(len(val_dataset)):
        token_to_dataset_idx[val_dataset.data_infos[idx]['token']] = idx

    # Group by scene
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=cfg.data.val.data_root, verbose=False)
    from nuscenes.utils.splits import val as VAL_SCENES
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

    log.info('Tracking eval: %d scenes, %d samples',
             len(scene_samples), sum(len(v) for v in scene_samples.values()))

    # Run tracking
    all_results = {}
    num_scenes = len(scene_samples)

    for si, (scene_token, samples) in enumerate(scene_samples.items()):
        scene = nusc.get('scene', scene_token)
        if (si + 1) % 20 == 0 or si == 0:
            log.info('[%d/%d] %s (%d frames)', si+1, num_scenes,
                     scene['name'], len(samples))
        tracker.reset()
        inner.prev_scene_token = None
        inner.pts_bbox_head.reset_memory()

        for _, sample_token in samples:
            didx = token_to_dataset_idx.get(sample_token)
            if didx is None:
                continue

            data = val_dataset[didx]
            img = data['img'][0].data.unsqueeze(0).cuda()
            img_metas = [data['img_metas'][0].data]

            with torch.no_grad():
                data_dict = {}
                for k in ['lidar2img', 'intrinsics', 'extrinsics',
                          'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']:
                    if k in data:
                        val = data[k][0] if isinstance(data[k], list) else data[k]
                        if hasattr(val, 'data'):
                            val = val.data
                        if isinstance(val, torch.Tensor):
                            data_dict[k] = val.unsqueeze(0).cuda()
                        else:
                            data_dict[k] = val

                data_dict['img'] = img
                data_dict['img_feats'] = inner.extract_img_feat(img, 1)

                # Scene change detection
                scene_tok = img_metas[0].get('scene_token',
                    token_to_info[sample_token].get('scene_token', ''))
                if scene_tok != inner.prev_scene_token:
                    inner.prev_scene_token = scene_tok
                    data_dict['prev_exists'] = img.new_zeros(1)
                    inner.pts_bbox_head.reset_memory()
                else:
                    data_dict['prev_exists'] = img.new_ones(1)

                from projects.mmdet3d_plugin.models.utils.misc import locations
                pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
                bs, n = data_dict['img_feats'].shape[:2]
                x = data_dict['img_feats'].flatten(0, 1)
                location = locations(x, inner.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)

                outs = inner.pts_bbox_head(
                    location, img_metas, None, **data_dict)

                # ego_pose: [B, 4, 4] lidar2global
                cur_ego_pose = data_dict['ego_pose'][0]  # [4, 4]
                frame_results = tracker.track_frame(outs, img_metas, cur_ego_pose)
                all_results[sample_token] = frame_results

    # Build submission
    log.info('Building submission JSON...')
    submission = {
        'meta': {'use_camera': True, 'use_lidar': False,
                 'use_radar': False, 'use_map': False,
                 'use_external': False},
        'results': defaultdict(list)
    }

    for sample_token, frame_results in all_results.items():
        info = token_to_info[sample_token]
        for det in frame_results:
            bbox = det['bbox'].cpu().numpy().astype(float)
            class_name = CLASS_NAMES[det['label']]
            if class_name not in TRACKING_NAMES:
                continue
            pos, size, rot_quat, vel = lidar_to_global(bbox, info)
            submission['results'][sample_token].append({
                'sample_token': sample_token,
                'translation': pos,
                'size': size,
                'rotation': rot_quat.elements.tolist(),
                'velocity': vel,
                'tracking_id': str(det['tracking_id']),
                'tracking_name': class_name,
                'tracking_score': det['score'],
                'attribute_name': ATTR_MAP.get(class_name, ''),
            })

    # Ensure all val samples have entries
    for scene_token, samples in scene_samples.items():
        for _, sample_token in samples:
            if sample_token not in submission['results']:
                submission['results'][sample_token] = []

    out_path = os.path.join(os.path.dirname(args.checkpoint), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(submission, f)
    log.info('Saved: %s', out_path)

    # Run nuScenes tracking eval
    log.info('Running nuScenes TrackingEval...')
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

    if isinstance(result, dict):
        summary = result
    else:
        summary = result.serialize()

    log.info('=== Tracking Results ===')
    label_metrics = summary.get('label_metrics', {})
    for metric_name in ['amota', 'amotp', 'recall', 'mota', 'motp', 'ids', 'frag']:
        if metric_name in label_metrics:
            vals = label_metrics[metric_name]
            log.info('  %s: %.4f', metric_name,
                     float(np.nanmean(list(vals.values()))))
    log.info('Done.')


if __name__ == '__main__':
    main()
