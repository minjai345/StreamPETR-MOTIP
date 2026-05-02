"""Oracle association experiment: replace predicted tracking_id with
stable GT-derived ID for detections matched to GT.

Purpose: measure how many IDS come from UNMATCHED detections alone.

    IDS ≈ baseline → detection ceiling dominates, association already OK
    IDS ≪ baseline → association is the bottleneck

Only output tracking_id is swapped — context, decoder, and all tracker
internals are untouched. So this is not a fix for exposure bias; it's a
diagnostic that isolates detection failure from association failure.

Usage:
    python tools/experiments/oracle_association_eval.py \
        --config <CONFIG> --checkpoint <CKPT> --feats <FEATS.pkl> \
        --out tracking_results_oracle.json
"""
import os, sys, json, argparse, pickle, logging
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--det-thresh', type=float, default=0.25)
    ap.add_argument('--new-thresh', type=float, default=0.40)
    ap.add_argument('--id-thresh', type=float, default=0.10)
    ap.add_argument('--max-age', type=int, default=5)
    ap.add_argument('--match-dist', type=float, default=2.0,
                    help='Center distance (m) threshold for GT match')
    ap.add_argument('--out', default='tracking_results_oracle.json')
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

    mc = dict(cfg.model.motip_cfg)
    mc['det_thresh'] = args.det_thresh
    mc['new_thresh'] = args.new_thresh
    mc['id_thresh'] = args.id_thresh
    mc['max_age'] = args.max_age
    tracker = MOTIPTracker(model, mc)

    log.info('Running oracle-association tracker: %d scenes', len(scene_samples))
    all_results = {}  # tok → list of results with possibly-replaced tracking_id
    n_matched_total = 0
    n_results_total = 0
    for si, (scene_tok, samples) in enumerate(scene_samples.items()):
        tracker.reset()
        for _, tok in samples:
            if tok not in all_feats: continue
            fd = all_feats[tok]
            results = tracker.track_frame(fd, fd['ego_pose'])
            n_results_total += len(results)

            if not results:
                all_results[tok] = []
                continue

            # Fetch GT for this sample (tracking classes only)
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

            # Convert predictions to global centers for matching
            pred_centers = []
            for r in results:
                bbox_np = r['bbox'].cpu().numpy().astype(float)
                pos, _, _, _ = lidar_to_global(bbox_np, info)
                pred_centers.append(pos[:2])
            pred_centers = np.array(pred_centers) if pred_centers else np.zeros((0,2))
            gt_centers = np.array(gt_centers) if gt_centers else np.zeros((0,2))

            # Greedy matching (nearest within match_dist)
            if len(pred_centers) > 0 and len(gt_centers) > 0:
                dists = np.linalg.norm(
                    pred_centers[:, None, :] - gt_centers[None, :, :], axis=-1)
                pred_to_gt = [-1] * len(results)
                used = set()
                for pi in np.argsort(dists.min(axis=1)):
                    if dists[pi].min() > args.match_dist: continue
                    for gi in np.argsort(dists[pi]):
                        if gi in used or dists[pi, gi] > args.match_dist: continue
                        pred_to_gt[pi] = gi
                        used.add(gi)
                        break
            else:
                pred_to_gt = [-1] * len(results)

            # Swap tracking_id with GT instance_token for matched
            for i, r in enumerate(results):
                gi = pred_to_gt[i]
                if gi >= 0:
                    r['tracking_id'] = 'gt_' + gt_inst[gi]
                    n_matched_total += 1
                # else: keep predicted numeric id (as str for JSON)
            all_results[tok] = results
        if (si + 1) % 30 == 0:
            log.info('  [%d/%d] matched so far: %d/%d (%.1f%%)',
                     si + 1, len(scene_samples), n_matched_total,
                     n_results_total, 100 * n_matched_total / max(n_results_total, 1))

    log.info('Total: matched %d / %d results (%.1f%%)',
             n_matched_total, n_results_total,
             100 * n_matched_total / max(n_results_total, 1))

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

    out_path = os.path.join(os.path.dirname(args.checkpoint), args.out)
    with open(out_path, 'w') as f:
        json.dump({'meta':{'use_camera':True,'use_lidar':False,'use_radar':False,
                  'use_map':False,'use_external':False},
                  'results':dict(submission)}, f)
    log.info('Saved submission: %s', out_path)

    eval_cfg = config_factory('tracking_nips_2019')
    te = TrackingEval(config=eval_cfg, result_path=out_path, eval_set='val',
                      output_dir='/tmp/oracle_out/', nusc_version='v1.0-trainval',
                      nusc_dataroot=cfg.data.val.data_root, verbose=False)
    res = te.main()
    summ = res if isinstance(res, dict) else res.serialize()
    lm = summ.get('label_metrics', {})
    amota = np.nanmean(list(lm['amota'].values())) if 'amota' in lm else 0
    mota = np.nanmean(list(lm['mota'].values())) if 'mota' in lm else 0
    ids = int(np.nansum(list(lm['ids'].values()))) if 'ids' in lm else 0
    log.info('=== ORACLE ASSOCIATION RESULTS ===')
    log.info(f'AMOTA={amota:.4f} MOTA={mota:.4f} IDS={ids}')


if __name__ == '__main__':
    main()
