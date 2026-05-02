"""GT-feeding experiment: measure exposure bias by overriding inference
context with GT-derived slot identities.

Normal inference (autoregressive):
    predicted slot → id_emb → context for next frame

GT-feeding (this script):
    GT instance_token → persistent slot (per-scene) → id_emb → context
    (tracking_id output = still predicted, for IDS measurement)

Interpretation:
    IDS drops sharply  → exposure bias is main cause (context corruption)
    IDS similar        → association itself is weak (motion/feature)
    IDS mid            → both contribute

Usage:
    python tools/gt_feeding_eval.py \
        --config <CONFIG> --checkpoint <CKPT> --feats <FEATS.pkl> \
        --det-thresh 0.25 --new-thresh 0.40 --id-thresh 0.10 \
        --out tracking_results_gtfed.json
"""
import os, sys, json, argparse, pickle, logging
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
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.tracking.evaluate import TrackingEval

from tools.eval_tracking import (MOTIPTracker, lidar_to_global,
                                  CLASS_NAMES, TRACKING_NAMES, ATTR_MAP)


class GTFeedingTracker(MOTIPTracker):
    """MOTIPTracker that overrides history id_emb with GT-derived slots.

    Per scene, maintains instance_token → slot map so the same GT object
    always produces the same id_emb in context.
    """

    def reset_scene_gt(self):
        self.gt_inst_to_slot = {}
        self.gt_slot_pool = list(range(self.K))  # available slot pool

    def set_frame_gt(self, gt_inst_tokens, gt_centers_global):
        """Record GT for upcoming track_frame call."""
        self._frame_gt_inst = list(gt_inst_tokens)
        self._frame_gt_pts = (np.array(gt_centers_global)
                              if len(gt_centers_global) else
                              np.zeros((0, 2)))

    def _gt_slot_for_instance(self, inst_tok):
        if inst_tok in self.gt_inst_to_slot:
            return self.gt_inst_to_slot[inst_tok]
        if not self.gt_slot_pool:
            return None
        slot = self.gt_slot_pool.pop(0)
        self.gt_inst_to_slot[inst_tok] = slot
        return slot

    def track_frame(self, sample_data, ego_pose):
        results = super().track_frame(sample_data, ego_pose)

        # After super().track_frame, self.history[-1] has the autoregressive
        # id_emb. Override it with GT-derived slots where predictions match GT.
        if not results or len(self.history) == 0:
            return results
        if not hasattr(self, '_frame_gt_inst') or len(self._frame_gt_inst) == 0:
            return results

        device = results[0]['bbox'].device
        m = self.model.module if hasattr(self.model, 'module') else self.model

        # Convert each result's LiDAR-frame bbox to global center for matching
        info = self._frame_info
        pred_centers_global = []
        for r in results:
            bbox_np = r['bbox'].cpu().numpy().astype(float)
            pos, _, _, _ = lidar_to_global(bbox_np, info)
            pred_centers_global.append(pos[:2])
        pred_centers_global = np.array(pred_centers_global)
        gt_centers = self._frame_gt_pts

        n_pred = len(results)
        n_gt = len(gt_centers)
        if n_pred == 0 or n_gt == 0:
            return results

        dists = np.linalg.norm(
            pred_centers_global[:, None, :] - gt_centers[None, :, :], axis=-1)
        # Greedy matching: for each pred, best gt within 2m
        match_thresh = 2.0
        pred_to_gt = [-1] * n_pred
        used_gt = set()
        for pi in np.argsort(dists.min(axis=1)):
            if dists[pi].min() > match_thresh:
                continue
            for gi in np.argsort(dists[pi]):
                if gi in used_gt or dists[pi, gi] > match_thresh:
                    continue
                pred_to_gt[pi] = gi
                used_gt.add(gi)
                break

        # Build override slot_ids: GT slot for matched, predicted slot for unmatched
        h = self.history[-1]
        override_slots = []
        any_override = False
        for i, r in enumerate(results):
            gi = pred_to_gt[i]
            if gi >= 0:
                inst_tok = self._frame_gt_inst[gi]
                gt_slot = self._gt_slot_for_instance(inst_tok)
                if gt_slot is not None:
                    override_slots.append(gt_slot)
                    any_override = True
                    continue
            # fallback: keep predicted slot
            override_slots.append(self.tracks[r['tracking_id']]['id_slot'])

        if any_override:
            slot_ids = torch.tensor(override_slots, device=device, dtype=torch.long)
            new_id_emb = m.id_dict.get_id_embedding(slot_ids)
            h['id_emb'] = new_id_emb.clone()

        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--det-thresh', type=float, default=0.25)
    ap.add_argument('--new-thresh', type=float, default=0.40)
    ap.add_argument('--id-thresh', type=float, default=0.10)
    ap.add_argument('--max-age', type=int, default=5)
    ap.add_argument('--out', default='tracking_results_gtfed.json')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    import importlib; importlib.import_module('projects.mmdet3d_plugin')

    model = build_model(cfg.model)
    wrap_fp16_model(model)
    model.cuda().eval()
    load_checkpoint(model, args.checkpoint, map_location='cuda', strict=False)

    log.info('Loading features: %s', args.feats)
    raw_outputs = pickle.load(open(args.feats, 'rb'))

    val_cfg = cfg.data.val.copy(); val_cfg['test_mode'] = True
    dataset = build_dataset(val_cfg)
    data_infos = dataset.data_infos
    val_data = pickle.load(open(cfg.data.val.ann_file, 'rb'))
    token_to_info = {info['token']: info for info in val_data['infos']}

    # Convert features to token-keyed (top-300 per sample)
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    pc_range = [-51.2,-51.2,-5,51.2,51.2,3]
    all_feats = {}
    for idx, out in enumerate(raw_outputs):
        info = data_infos[idx]; tok = info['token']
        pts = out['pts_bbox']
        cls_sig = pts['cls_scores'].sigmoid()
        scores_flat, indexs = cls_sig.view(-1).topk(300)
        num_classes = pts['cls_scores'].shape[1]
        labels = indexs % num_classes
        qi = torch.div(indexs, num_classes, rounding_mode='trunc')
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
    del raw_outputs

    nusc = NuScenes(version='v1.0-trainval',
                    dataroot=cfg.data.val.data_root, verbose=False)
    val_scene_names = set(VAL_SCENES)
    scene_samples = defaultdict(list)
    for info in val_data['infos']:
        sample = nusc.get('sample', info['token'])
        scene = nusc.get('scene', sample['scene_token'])
        if scene['name'] in val_scene_names:
            scene_samples[sample['scene_token']].append((info['timestamp'], info['token']))
    for st in scene_samples: scene_samples[st].sort()

    # Tracker setup
    mc = dict(cfg.model.motip_cfg)
    mc['det_thresh'] = args.det_thresh
    mc['new_thresh'] = args.new_thresh
    mc['id_thresh'] = args.id_thresh
    mc['max_age'] = args.max_age
    tracker = GTFeedingTracker(model, mc)

    log.info('Running GT-feeding tracker: %d scenes', len(scene_samples))
    all_results = {}
    for si, (scene_tok, samples) in enumerate(scene_samples.items()):
        tracker.reset()
        tracker.reset_scene_gt()
        for _, tok in samples:
            if tok not in all_feats: continue
            fd = all_feats[tok]
            ego = fd['ego_pose']

            # Fetch GT for this sample (tracking classes only)
            sample = nusc.get('sample', tok)
            gt_inst, gt_centers = [], []
            for ann_tok in sample['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                cat = ann['category_name']
                matched = False
                for n in ['pedestrian','car','truck','bus','trailer','motorcycle','bicycle']:
                    if n in cat: matched = True; break
                if matched:
                    gt_inst.append(ann['instance_token'])
                    gt_centers.append(ann['translation'][:2])
            tracker.set_frame_gt(gt_inst, gt_centers)
            tracker._frame_info = token_to_info[tok]
            frame_results = tracker.track_frame(fd, ego)
            all_results[tok] = frame_results
        if (si + 1) % 20 == 0:
            log.info('  [%d/%d] done', si + 1, len(scene_samples))

    # Build submission (same as eval_tracking.py)
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

    out_path = os.path.join(os.path.dirname(args.checkpoint), args.out)
    with open(out_path, 'w') as f:
        json.dump({'meta':{'use_camera':True,'use_lidar':False,'use_radar':False,
                  'use_map':False,'use_external':False},
                  'results':dict(submission)}, f)
    log.info('Saved submission: %s', out_path)

    # nuScenes TrackingEval
    eval_cfg = config_factory('tracking_nips_2019')
    te = TrackingEval(config=eval_cfg, result_path=out_path, eval_set='val',
                      output_dir='/tmp/gtfed_out/', nusc_version='v1.0-trainval',
                      nusc_dataroot=cfg.data.val.data_root, verbose=False)
    res = te.main()
    summ = res if isinstance(res, dict) else res.serialize()
    lm = summ.get('label_metrics', {})
    amota = np.nanmean(list(lm['amota'].values())) if 'amota' in lm else 0
    mota = np.nanmean(list(lm['mota'].values())) if 'mota' in lm else 0
    ids = int(np.nansum(list(lm['ids'].values()))) if 'ids' in lm else 0
    log.info('=== GT-FEEDING RESULTS ===')
    log.info(f'AMOTA={amota:.4f} MOTA={mota:.4f} IDS={ids}')


if __name__ == '__main__':
    main()
