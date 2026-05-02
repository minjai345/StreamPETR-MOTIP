"""Pedestrian same_class_swap proximity AND query-feature similarity diagnostic.

For each same_class_swap event for class C (pedestrian default):
  1. Identify victim GT P (gt_token in event)
  2. Find P' = "original owner" of new_pred_id before this event
  3. Find common frame where both P and P' were matched (else closest)
  4. Compute:
     - 3D distance(P, P') at common frame
     - cosine_similarity(query_feat[P's matched pred], query_feat[P''s matched pred])
  5. 2D distribution → diagnose

Interpretation table:
  near + similar     → both motion + contrastive viable
  near + dissimilar  → proximity is dominant → motion cue prescription
  far  + similar     → feature similarity dominant → contrastive prescription
  far  + dissimilar  → other cause (context corruption, etc.)

Usage:
    python tools/experiments/swap_proximity_analysis.py \\
        --events <ids_events.jsonl> \\
        --results <baseline.json> \\
        --feats <track_feats.pkl> \\
        --cls pedestrian
"""
import os, sys, json, argparse, logging, pickle
from collections import defaultdict, Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
import torch
from pyquaternion import Quaternion as Q
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES

from tools.experiments.ids_event_decompose import (
    load_gt_trajectories, load_preds, per_class_match, TRACKING_NAMES, CATEGORY_MAP,
)


PC_RANGE = [-51.2, -51.2, -5, 51.2, 51.2, 3]


def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--feats', required=True, help='track_feats_iter*.pkl')
    ap.add_argument('--cls', default='pedestrian')
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--config', required=True,
                    help='Same config used to extract feats (for matching dataset.data_infos order)')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.events)
    os.makedirs(out_dir, exist_ok=True)

    log.info('Loading events from %s', args.events)
    target_events = []
    with open(args.events) as f:
        for line in f:
            e = json.loads(line)
            if e['gt_class'] == args.cls and e['category'] == 'same_class_swap':
                target_events.append(e)
    log.info('  %d %s same_class_swap events', len(target_events), args.cls)

    log.info('Loading NuScenes API...')
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    log.info('Loading GT trajectories + predictions...')
    gt_traj = load_gt_trajectories(nusc, None)
    submission = json.load(open(args.results))
    preds_by_class, _, _ = load_preds(submission, nusc)

    log.info('Per-class matching for class %s...', args.cls)
    matched_per_scene = {}
    pred_owner = defaultdict(lambda: defaultdict(list))
    inst_pos = defaultdict(dict)
    for st, instances in gt_traj.items():
        cls_insts = {inst: seq for inst, seq in instances.items() if seq[0][3] == args.cls}
        if not cls_insts: continue
        n_frames = max(idx for seq in instances.values() for (idx,_,_,_) in seq) + 1
        preds_cls = preds_by_class.get(st, {}).get(args.cls, {})
        m = per_class_match(cls_insts, preds_cls, n_frames)
        matched_per_scene[st] = m
        for inst, seq in cls_insts.items():
            inst_pos[st][inst] = {}
            for (idx, _, pos, _) in seq:
                inst_pos[st][inst][idx] = pos
        for inst, mseq in m.items():
            for (fidx, pid, _) in mseq:
                if pid is not None:
                    pred_owner[st][pid].append((fidx, inst))

    # Build sample_tok → (scene_tok, fidx) map
    sample_to_scene = {}
    sample_at = {}  # (scene_tok, fidx) → sample_tok
    for scene in nusc.scene:
        st = scene['token']
        s = scene['first_sample_token']
        idx = 0
        while s:
            sample_to_scene[s] = (st, idx)
            sample_at[(st, idx)] = s
            sample = nusc.get('sample', s)
            s = sample['next']
            idx += 1

    # Build per-sample submission map: (sample_tok, tracking_id) → translation
    sub_tid_pos = defaultdict(dict)
    for sample_tok, dets in submission['results'].items():
        for det in dets:
            tid = str(det['tracking_id'])
            sub_tid_pos[sample_tok][tid] = np.array(det['translation'][:2])

    log.info('Loading features pickle (%s, ~70s)...', args.feats)
    raw_outputs = pickle.load(open(args.feats, 'rb'))
    log.info('  loaded %d sample outputs', len(raw_outputs))

    log.info('Loading dataset for data_infos (same order as feats extraction)...')
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(args.config)
    import importlib; importlib.import_module('projects.mmdet3d_plugin')
    val_cfg = cfg.data.val.copy(); val_cfg['test_mode'] = True
    dataset = build_dataset(val_cfg)
    data_infos = dataset.data_infos
    log.info('  data_infos: %d (raw_outputs: %d)', len(data_infos), len(raw_outputs))

    # Build sample_tok → (top300 global pos, label, query_feat) — only ped class
    log.info('Building per-sample feature index (ped class)...')
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    PED_LABEL = TRACKING_NAMES.index('pedestrian') if args.cls == 'pedestrian' else None

    # Resolve target class label index by matching CLASS_NAMES used in tracker
    # tracker labels = nuScenes 10-class detection order (use first class match in CATEGORY_MAP)
    # Here we just match by tracking_name string from submission instead. But for top-300
    # we need the integer label. Use CLASS_NAMES from eval_tracking.
    from tools.eval_tracking import CLASS_NAMES
    cls_label = CLASS_NAMES.index(args.cls)

    from tools.eval_tracking import lidar_to_global

    sample_feats = {}  # sample_tok → (positions [N,2] global, feats [N,C])
    for idx, out in enumerate(raw_outputs):
        info = data_infos[idx]
        tok = info['token']
        pts = out['pts_bbox']
        cls_sig = pts['cls_scores'].sigmoid()
        scores_flat, indexs = cls_sig.view(-1).topk(300)
        num_classes = pts['cls_scores'].shape[1]
        labels = indexs % num_classes
        qi = torch.div(indexs, num_classes, rounding_mode='trunc')
        mask = labels == cls_label
        if mask.sum() == 0:
            sample_feats[tok] = (np.zeros((0, 2)), np.zeros((0, pts['query_feat'].shape[-1])))
            continue
        qi_f = qi[mask]
        decoded = denormalize_bbox(pts['bbox_raw'][qi_f], PC_RANGE).numpy()
        feats = pts['query_feat'][qi_f].numpy()
        # Use SAME transform as tracker (lidar_to_global on each row)
        positions_global = np.zeros((decoded.shape[0], 2))
        for i in range(decoded.shape[0]):
            pos, _, _, _ = lidar_to_global(decoded[i].astype(float), info)
            positions_global[i] = pos[:2]
        sample_feats[tok] = (positions_global, feats)
    del raw_outputs
    log.info('  feature index built for %d samples', len(sample_feats))

    def get_feat(sample_tok, target_pos, max_dist=5.0):
        """Lookup query_feat for the top-300 ped det closest to target_pos."""
        if sample_tok not in sample_feats: return None, None
        positions, feats = sample_feats[sample_tok]
        if len(positions) == 0: return None, None
        d = np.linalg.norm(positions - target_pos[None, :], axis=1)
        best = np.argmin(d)
        if d[best] > max_dist: return None, float(d[best])
        return feats[best], float(d[best])

    # Debug: check ped count per sample
    ped_counts = [len(p) for p, _ in sample_feats.values()]
    log.info('Ped top-300 counts per sample: median=%d, mean=%.1f, max=%d, min=%d, zero_samples=%d',
             int(np.median(ped_counts)), np.mean(ped_counts), max(ped_counts),
             min(ped_counts), sum(1 for c in ped_counts if c == 0))

    log.info('Analyzing %d events...', len(target_events))
    results = []
    skip_no_owner = 0
    skip_no_common = 0
    skip_no_pos = 0
    skip_no_feat = 0

    for ev in target_events:
        st = ev['scene']
        P = ev['gt_token']
        new_Y = ev['new_pred_id']
        f_new = ev['frame_new']

        owners = pred_owner.get(st, {}).get(new_Y, [])
        prior = [(f, inst) for (f, inst) in owners if f < f_new and inst != P]
        if not prior:
            skip_no_owner += 1; continue
        prior.sort(key=lambda x: -x[0])
        f_prior_last, P_prime = prior[0]

        # Find common frame where BOTH P and P' were matched (preferred)
        # Try f_prior_last first, then earlier
        common_frame = None
        for cf in sorted(set(inst_pos[st].get(P, {}).keys()) &
                         set(inst_pos[st].get(P_prime, {}).keys()),
                         reverse=True):
            if cf > f_new: continue
            # Check that both were matched (have pred_id) at cf
            P_matched = any(fidx == cf and pid is not None
                            for (fidx, pid, _) in matched_per_scene.get(st, {}).get(P, []))
            Pp_matched = any(fidx == cf and pid is not None
                             for (fidx, pid, _) in matched_per_scene.get(st, {}).get(P_prime, []))
            if P_matched and Pp_matched:
                common_frame = cf
                break
        if common_frame is None:
            skip_no_common += 1; continue

        # Positions at common_frame
        P_pos = np.array(inst_pos[st][P][common_frame])
        Pp_pos = np.array(inst_pos[st][P_prime][common_frame])
        dist_3d = float(np.linalg.norm(P_pos - Pp_pos))

        # Get matched pred ids at common_frame
        P_pid = next((pid for (f, pid, _) in matched_per_scene[st][P] if f == common_frame), None)
        Pp_pid = next((pid for (f, pid, _) in matched_per_scene[st][P_prime] if f == common_frame), None)
        sample_tok = sample_at.get((st, common_frame))
        if sample_tok is None:
            skip_no_pos += 1; continue

        # Get submission positions for these tids
        P_sub_pos = sub_tid_pos.get(sample_tok, {}).get(P_pid)
        Pp_sub_pos = sub_tid_pos.get(sample_tok, {}).get(Pp_pid)
        if P_sub_pos is None or Pp_sub_pos is None:
            skip_no_pos += 1; continue

        P_feat, P_fd = get_feat(sample_tok, P_sub_pos)
        Pp_feat, Pp_fd = get_feat(sample_tok, Pp_sub_pos)
        if P_feat is None or Pp_feat is None:
            skip_no_feat += 1
            if skip_no_feat <= 5:
                log.info('  no_feat eg: P_pos=%s, P_min_dist=%s, Pp_pos=%s, Pp_min_dist=%s',
                         P_sub_pos.tolist(), P_fd, Pp_sub_pos.tolist(), Pp_fd)
            continue

        cos = cosine(P_feat, Pp_feat)
        results.append({
            'scene': st, 'P': P, 'P_prime': P_prime,
            'frame_new': f_new, 'common_frame': common_frame,
            'dist_3d_m': dist_3d, 'cos_sim': cos,
        })

    log.info('=== Analysis ===')
    log.info('Total events:          %d', len(target_events))
    log.info('Analyzed:              %d', len(results))
    log.info('Skipped no_owner:      %d', skip_no_owner)
    log.info('Skipped no_common_frame:%d', skip_no_common)
    log.info('Skipped no_pos:        %d', skip_no_pos)
    log.info('Skipped no_feat:       %d', skip_no_feat)

    if not results:
        log.warning('No analyzable events.'); return

    dists = np.array([r['dist_3d_m'] for r in results])
    sims = np.array([r['cos_sim'] for r in results])

    log.info('')
    log.info('Distance (m):    median=%.2f mean=%.2f max=%.2f',
             np.median(dists), np.mean(dists), np.max(dists))
    log.info('Cos sim:         median=%.3f mean=%.3f', np.median(sims), np.mean(sims))

    # 2D quadrant analysis
    NEAR_TH = 3.0   # within 3m
    SIM_TH = 0.9    # cos sim > 0.9 = "very similar"
    near_sim = ((dists <= NEAR_TH) & (sims >= SIM_TH)).sum()
    near_dis = ((dists <= NEAR_TH) & (sims < SIM_TH)).sum()
    far_sim  = ((dists >  NEAR_TH) & (sims >= SIM_TH)).sum()
    far_dis  = ((dists >  NEAR_TH) & (sims < SIM_TH)).sum()
    n = len(results)
    log.info('')
    log.info('Quadrants (NEAR<=%.0fm, SIM>=%.2f):', NEAR_TH, SIM_TH)
    log.info('  NEAR + SIMILAR  : %d (%.1f%%)  [proximity AND feature]', near_sim, 100*near_sim/n)
    log.info('  NEAR + DISSIM   : %d (%.1f%%)  [proximity dominant → motion cue]', near_dis, 100*near_dis/n)
    log.info('  FAR  + SIMILAR  : %d (%.1f%%)  [feature dominant → contrastive]', far_sim, 100*far_sim/n)
    log.info('  FAR  + DISSIM   : %d (%.1f%%)  [other cause]', far_dis, 100*far_dis/n)

    # Distance histogram
    dbins = [0, 1, 2, 3, 5, 10, 20, 50, 1e6]
    dlabels = ['0-1m','1-2m','2-3m','3-5m','5-10m','10-20m','20-50m','>50m']
    dcounts, _ = np.histogram(dists, bins=dbins)
    log.info('')
    log.info('Distance distribution:')
    cum = 0
    for lbl, c in zip(dlabels, dcounts):
        cum += c
        log.info('  %-8s %5d (%.1f%%)  [cum %.1f%%]', lbl, c, 100*c/n, 100*cum/n)

    sbins = [-1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
    slabels = ['<0.5','0.5-0.7','0.7-0.8','0.8-0.9','0.9-0.95','0.95-0.99','>0.99']
    scounts, _ = np.histogram(sims, bins=sbins)
    log.info('')
    log.info('Cos-sim distribution:')
    cum = 0
    for lbl, c in zip(slabels, scounts):
        cum += c
        log.info('  %-9s %5d (%.1f%%)  [cum %.1f%%]', lbl, c, 100*c/n, 100*cum/n)

    out_path = os.path.join(out_dir, f'swap_diag_{args.cls}.jsonl')
    with open(out_path, 'w') as f:
        for r in results: f.write(json.dumps(r) + '\n')
    log.info('Saved: %s', out_path)


if __name__ == '__main__':
    main()
