"""GT-aligned feeding (proper redesign v2):
slot_to_global is mutated ONLY by GT registration (matched dets) and
unmatched-FP newborn allocation. Matched dets never spawn extra slots.

Exposure bias isolate:
  - history id_emb = GT-slot based (clean context)
  - slot_to_global = GT gid 로 aligned (decoder prediction이 의미있게 해석됨)
  - decoder forward 그대로 실행
  - matched + decoder-newborn: 출력 tracking_id만 fresh gid (slot_to_global 미수정) →
    nuScenes eval에서 IDS 카운트되지만 slot pool 잠식 없음. structural failure 측정 깨끗.
  - GT 관측 = self.tracks[gid_I] age=0 강제 → GT 관측 중인 slot은 절대 evict 안 됨.
  - _free_slot 시 inst_to_gid 동시 청소 → stale registration 차단.

Per-outcome counters:
  matched_correct       : matched, decoder picked GT slot
  matched_wrong_slot    : matched, decoder picked different existing slot (ID swap)
  matched_newborn       : matched, decoder said newborn (structural failure)
  matched_dropped       : matched, decoder uncertain (score/prob below thresholds)
  unmatched_existing    : FP aliased to existing track
  unmatched_newborn     : FP allocated new track
  unmatched_dropped     : FP dropped

Interpretation (defaults same as config 3 baseline: IDS=2547, oracle_assoc IDS=256):
  IDS ≪ 2547            → exposure bias 주원인
  IDS ≈ 2547            → decoder 구조적 한계 (motion cue 등)
  matched_wrong + matched_newborn ≈ structural ID swap count

Usage:
    python tools/experiments/gt_aligned_eval.py \\
        --config <C> --checkpoint <CK> --feats <F.pkl> \\
        --out-dir work_dirs/.../gt_aligned_strict/
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


class GTAlignedTracker(MOTIPTracker):
    """Tracker with slot-space aligned to GT instance_tokens.

    Public hooks:
      - set_frame_gt(inst_tokens, centers_global): provide current frame's GT
      - set_frame_info(info): nuScenes info dict for coord conversion
    """

    # Keys for per-outcome bookkeeping
    COUNT_KEYS = [
        'matched_correct', 'matched_wrong_slot',
        'matched_newborn', 'matched_dropped',
        'unmatched_existing', 'unmatched_newborn', 'unmatched_dropped',
        'first_frame_matched', 'first_frame_unmatched',
        'matched_pool_full',
    ]

    def reset(self):
        super().reset()
        # Per-scene GT state
        self.inst_to_gid = {}        # instance_token → persistent gid
        self.inst_to_slot = {}       # instance_token → persistent slot
        self.gid_to_inst = {}        # gid → instance_token (reverse, for stale cleanup)
        self._frame_gt_inst = []
        self._frame_gt_pts = np.zeros((0, 2))
        self._frame_info = None
        self.counts = {k: 0 for k in self.COUNT_KEYS}

    def _free_slot(self, global_id):
        """Override: also drop stale GT registration for this gid."""
        super()._free_slot(global_id)
        if global_id in self.gid_to_inst:
            inst_tok = self.gid_to_inst.pop(global_id)
            self.inst_to_gid.pop(inst_tok, None)
            self.inst_to_slot.pop(inst_tok, None)

    def set_frame_gt(self, inst_tokens, centers_global):
        self._frame_gt_inst = list(inst_tokens)
        self._frame_gt_pts = (np.array(centers_global, dtype=float)
                              if len(centers_global) else np.zeros((0, 2)))

    def set_frame_info(self, info):
        self._frame_info = info

    def _register_gt_instance(self, inst_tok):
        """Get or create persistent (gid, slot) for this GT instance.

        Updates slot_to_global so decoder-predicted GT slots resolve to
        the correct gid. Returns (gid, slot) or (None, None) if slot pool
        exhausted (K reached for this scene).
        """
        if inst_tok in self.inst_to_gid:
            return self.inst_to_gid[inst_tok], self.inst_to_slot[inst_tok]

        # Allocate a new GT-dedicated slot (same pool as normal tracker)
        device = 'cuda'
        # Find free slot
        free_slot = None
        for s in range(self.K):
            if s not in self.slot_to_global:
                free_slot = s
                break
        if free_slot is None:
            # slot pool full — skip GT registration (rare, scene has >K active)
            return None, None

        gid = self.next_global_id
        self.next_global_id += 1
        self.inst_to_gid[inst_tok] = gid
        self.inst_to_slot[inst_tok] = free_slot
        self.gid_to_inst[gid] = inst_tok
        self.slot_to_global[free_slot] = gid
        self.global_to_slot[gid] = free_slot
        return gid, free_slot

    @torch.no_grad()
    def track_frame(self, sample_data, ego_pose):
        """Like parent but:
          (a) register GT instances AS THEY ARE OBSERVED (by matching pred→GT),
          (b) override result tracking_ids for matched to GT gid (so tracking is consistent),
          (c) rewrite history[-1] id_emb using GT slots for matched detections.

        Decoder forward runs normally — predictions stay the decoder's own; we
        only align the slot-space so those predictions resolve through a clean
        slot_to_global mapping.
        """
        m = self.model.module if hasattr(self.model, 'module') else self.model
        device = next(m.parameters()).device

        scores = sample_data['scores'].to(device)
        labels = sample_data['labels'].to(device)
        bbox_decoded = sample_data['bbox_decoded'].to(device)
        bbox_raw = sample_data['bbox_raw'].to(device)
        query_feat = sample_data['query_feat'].to(device)
        ego_pose = ego_pose.to(device)

        # 1) score filter (same as parent)
        mask = scores > self.det_thresh
        scores_f = scores[mask]
        labels_f = labels[mask]
        bbox_decoded_f = bbox_decoded[mask]
        bbox_raw_f = bbox_raw[mask]
        det_feat = query_feat[mask]
        N_det = len(scores_f)

        if N_det == 0:
            dead = [g for g, t in self.tracks.items()
                    if (t['age'] + 1) > self.max_age]
            for g in list(self.tracks.keys()):
                self.tracks[g]['age'] += 1
            for g in dead:
                self._free_slot(g)
                del self.tracks[g]
            return []

        pe_input = m._bbox_to_pe_input(bbox_raw_f)
        pe = m.pe_3d(pe_input[:, :7], pe_input[:, 7:9])

        # 2) Match predictions to GT (center distance, 2m)
        pred_to_gt = [-1] * N_det
        if len(self._frame_gt_inst) > 0 and self._frame_info is not None:
            pred_centers = np.zeros((N_det, 2))
            for i in range(N_det):
                bbox_np = bbox_decoded_f[i].cpu().numpy().astype(float)
                pos, _, _, _ = lidar_to_global(bbox_np, self._frame_info)
                pred_centers[i] = pos[:2]
            gt_centers = self._frame_gt_pts
            if len(gt_centers) > 0:
                d = np.linalg.norm(
                    pred_centers[:, None, :] - gt_centers[None, :, :], axis=-1)
                used = set()
                for pi in np.argsort(d.min(axis=1)):
                    if d[pi].min() > 2.0:
                        continue
                    for gi in np.argsort(d[pi]):
                        if gi in used or d[pi, gi] > 2.0:
                            continue
                        pred_to_gt[pi] = gi
                        used.add(gi)
                        break

        # 3) Register GT instances for matched detections → populates
        #    slot_to_global BEFORE decoder runs. Returns per-detection
        #    (gt_gid, gt_slot) or (None, None) if unmatched or pool full.
        det_gt_info = [(None, None)] * N_det
        for i in range(N_det):
            gi = pred_to_gt[i]
            if gi >= 0:
                inst_tok = self._frame_gt_inst[gi]
                gid, slot = self._register_gt_instance(inst_tok)
                det_gt_info[i] = (gid, slot)
                if gid is None:
                    # Matched to GT but pool full → treated as unmatched.
                    # Track count so unmatched_* doesn't absorb this silently.
                    self.counts['matched_pool_full'] += 1

        # 4) First-frame special case (no history)
        has_history = len(self.history) > 0
        if not has_history:
            results = []
            for i in range(N_det):
                gt_gid, gt_slot = det_gt_info[i]
                if gt_gid is not None:
                    # Matched: use GT-aligned gid/slot (already registered)
                    gid, slot = gt_gid, gt_slot
                    self.counts['first_frame_matched'] += 1
                else:
                    # Unmatched: allocate normally
                    gid = self.next_global_id
                    self.next_global_id += 1
                    slot = self._allocate_slot(gid)
                    self.counts['first_frame_unmatched'] += 1
                self.tracks[gid] = {
                    'id_slot': slot, 'age': 0,
                    'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(),
                }
                results.append({
                    'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(), 'tracking_id': gid,
                    '_det_idx': i,
                })

            slot_ids_for_history = []
            for r in results:
                i = r['_det_idx']
                gt_gid, gt_slot = det_gt_info[i]
                slot_for_hist = gt_slot if gt_slot is not None else \
                                self.tracks[r['tracking_id']]['id_slot']
                slot_ids_for_history.append(slot_for_hist)
            slot_ids_t = torch.tensor(slot_ids_for_history,
                                      device=device, dtype=torch.long)
            id_emb = m.id_dict.get_id_embedding(slot_ids_t)
            self.history.append({
                'feat': det_feat.clone(), 'bbox': bbox_raw_f.clone(),
                'id_emb': id_emb.clone(), 'ego_pose': ego_pose.clone(),
            })
            if len(self.history) > self.context_len:
                self.history.pop(0)
            return results

        # 5) Build context from history (history already has GT-aligned id_emb
        #    from prior frames' updates)
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

        # 6) Assignment (same as parent), but slot_to_global is GT-aligned
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

            gt_gid, gt_slot = det_gt_info[i]
            is_matched = gt_gid is not None

            if slot_has_track and best_prob > newborn_prob and best_prob > self.id_thresh:
                gid = self.slot_to_global[best_slot]
                assigned_slots.add(best_slot)
                if is_matched:
                    if best_slot == gt_slot:
                        self.counts['matched_correct'] += 1
                    else:
                        self.counts['matched_wrong_slot'] += 1
                else:
                    self.counts['unmatched_existing'] += 1
            elif scores_f[i].item() > self.new_thresh:
                # Newborn branch — split by matched/unmatched.
                # Matched + decoder-newborn: emit fresh gid for OUTPUT only.
                #   slot_to_global is NOT modified (GT slot ownership preserved
                #   for future frames where decoder may pick correctly).
                #   nuScenes eval will count this as ID switch vs prior frames'
                #   gid_I — that's the structural failure signal we want.
                # Unmatched + decoder-newborn: standard FP newborn (allocate slot).
                gid = self.next_global_id
                self.next_global_id += 1
                if is_matched:
                    best_slot = gt_slot  # buffer uses GT slot
                    self.counts['matched_newborn'] += 1
                else:
                    best_slot = self._allocate_slot(gid)
                    assigned_slots.add(best_slot)
                    self.counts['unmatched_newborn'] += 1
            else:
                if is_matched:
                    self.counts['matched_dropped'] += 1
                else:
                    self.counts['unmatched_dropped'] += 1
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

        # 7a) Touch GT-registered gids whose instance was observed this frame.
        # Any matched det (any outcome — correct/wrong/newborn/dropped) keeps
        # gid_I alive in self.tracks with age=0. Without this, gid_I that
        # decoder never directly picks would never enter self.tracks and
        # leak slot_to_global forever (the "ghost slot" problem).
        observed_gids = set()
        for i in range(N_det):
            gt_gid, gt_slot = det_gt_info[i]
            if gt_gid is None:
                continue
            observed_gids.add(gt_gid)
            if gt_gid not in self.tracks:
                self.tracks[gt_gid] = {
                    'id_slot': gt_slot, 'age': 0,
                    'bbox': bbox_decoded_f[i], 'score': scores_f[i].item(),
                    'label': labels_f[i].item(),
                }
            else:
                self.tracks[gt_gid]['age'] = 0

        # 7b) Age & evict tracks not refreshed this frame.
        matched_gids = {r['tracking_id'] for r in results} | observed_gids
        dead = []
        for gid, t in self.tracks.items():
            if gid not in matched_gids:
                t['age'] += 1
                if t['age'] > self.max_age:
                    dead.append(gid)
        for gid in dead:
            self._free_slot(gid)
            del self.tracks[gid]

        # 8) Rewrite history id_emb: for matched, use GT slot so next frame's
        #    context reflects true identities (NOT decoder's slot choice).
        if results:
            slot_ids_for_history = []
            feat_for_history = []
            bbox_for_history = []
            for r in results:
                i = r['_det_idx']
                gt_gid, gt_slot = det_gt_info[i]
                slot_for_hist = gt_slot if gt_slot is not None else \
                                self.tracks[r['tracking_id']]['id_slot']
                slot_ids_for_history.append(slot_for_hist)
                feat_for_history.append(det_feat[i])
                bbox_for_history.append(bbox_raw_f[i])
            slot_ids_t = torch.tensor(slot_ids_for_history,
                                      device=device, dtype=torch.long)
            id_emb = m.id_dict.get_id_embedding(slot_ids_t)
            self.history.append({
                'feat': torch.stack(feat_for_history).clone(),
                'bbox': torch.stack(bbox_for_history).clone(),
                'id_emb': id_emb.clone(), 'ego_pose': ego_pose.clone(),
            })
        if len(self.history) > self.context_len:
            self.history.pop(0)

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
    ap.add_argument('--out-dir', default=None,
                    help='output dir (default: <ckpt_dir>/gt_aligned_strict)')
    ap.add_argument('--out', default='tracking_results_gtaligned_strict.json')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint),
                                            'gt_aligned_strict')
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import importlib; importlib.import_module('projects.mmdet3d_plugin')
    model = build_model(cfg.model); wrap_fp16_model(model)
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
    tracker = GTAlignedTracker(model, mc)

    log.info('Running GT-aligned (strict) tracker: %d scenes', len(scene_samples))
    all_results = {}
    cum_counts = {k: 0 for k in GTAlignedTracker.COUNT_KEYS}
    per_scene_counts = {}
    for si, (scene_tok, samples) in enumerate(scene_samples.items()):
        tracker.reset()
        for _, tok in samples:
            if tok not in all_feats: continue
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

            all_results[tok] = tracker.track_frame(fd, fd['ego_pose'])
        per_scene_counts[scene_tok] = dict(tracker.counts)
        for k, v in tracker.counts.items():
            cum_counts[k] += v
        if (si + 1) % 30 == 0:
            log.info('  [%d/%d] scenes done', si + 1, len(scene_samples))

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

    out_path = os.path.join(out_dir, args.out)
    with open(out_path, 'w') as f:
        json.dump({'meta':{'use_camera':True,'use_lidar':False,'use_radar':False,
                  'use_map':False,'use_external':False},
                  'results':dict(submission)}, f)
    log.info('Saved submission: %s', out_path)

    counts_path = os.path.join(out_dir, 'gt_aligned_counts.json')
    with open(counts_path, 'w') as f:
        json.dump({
            'cumulative': cum_counts,
            'per_scene': per_scene_counts,
            'thresholds': {
                'det_thresh': args.det_thresh, 'new_thresh': args.new_thresh,
                'id_thresh': args.id_thresh, 'max_age': args.max_age,
            },
            'checkpoint': args.checkpoint, 'feats': args.feats,
        }, f, indent=2)
    log.info('Saved counts: %s', counts_path)

    eval_cfg = config_factory('tracking_nips_2019')
    nusc_out = os.path.join(out_dir, 'nusc_eval')
    os.makedirs(nusc_out, exist_ok=True)
    te = TrackingEval(config=eval_cfg, result_path=out_path, eval_set='val',
                      output_dir=nusc_out, nusc_version='v1.0-trainval',
                      nusc_dataroot=cfg.data.val.data_root, verbose=False)
    res = te.main()
    summ = res if isinstance(res, dict) else res.serialize()
    lm = summ.get('label_metrics', {})
    amota = np.nanmean(list(lm['amota'].values())) if 'amota' in lm else 0
    mota = np.nanmean(list(lm['mota'].values())) if 'mota' in lm else 0
    ids = int(np.nansum(list(lm['ids'].values()))) if 'ids' in lm else 0
    log.info('=== GT-ALIGNED (STRICT) RESULTS ===')
    log.info(f'AMOTA={amota:.4f} MOTA={mota:.4f} IDS={ids}')
    log.info('--- Per-outcome counters ---')
    for k in GTAlignedTracker.COUNT_KEYS:
        log.info(f'  {k:<22s} {cum_counts[k]}')
    matched_total = (cum_counts['matched_correct'] + cum_counts['matched_wrong_slot']
                     + cum_counts['matched_newborn'] + cum_counts['matched_dropped'])
    structural_fail = cum_counts['matched_wrong_slot'] + cum_counts['matched_newborn']
    if matched_total > 0:
        log.info(f'matched_total={matched_total}  '
                 f'structural_fail(wrong_slot+newborn)={structural_fail}  '
                 f'({100.0*structural_fail/matched_total:.1f}% of matched)')


if __name__ == '__main__':
    main()
