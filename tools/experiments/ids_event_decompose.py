"""IDS Event Decomposition (Hybrid v2).

Primary: nuScenes-style per-class Hungarian matching at 2m threshold.
Sums approximately to nuScenes IDS at the operating point (single threshold,
not the AMOTA-averaged 2547; expect ±수백 deviation — methodological caveat).

Categories (precedence order — first match wins):
  1. track_break_recover  : gap >= 2 frames (None entries) before re-match with
                            different pred_id
  2. newborn_fp           : new pred_id never seen earlier in this scene+class
  3. same_class_swap      : everything else (incl. gap=1 momentary noise)

Secondary diagnostics (separate counters, not part of IDS sum):
  - detection_fail              : GT trajectories with class-internal match
                                  rate < 50%
  - cross_class_misdetection    : detection_fail GT that has SOME pred of
                                  another class within 2m at >= 1 frame.

Per-event jsonl dump for drill-down.

Usage:
    python tools/experiments/ids_event_decompose.py \\
        --results <tracking_results.json> --label baseline \\
        --out-dir work_dirs/.../ids_decompose_baseline/
"""
import os, sys, json, argparse, logging
from collections import defaultdict, Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES


TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
DIST_TH = 2.0  # nuScenes tracking_nips_2019.json dist_th_tp

CATEGORY_MAP = {
    'vehicle.car': 'car', 'vehicle.truck': 'truck', 'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus', 'vehicle.trailer': 'trailer',
    'vehicle.motorcycle': 'motorcycle', 'vehicle.bicycle': 'bicycle',
    'human.pedestrian.adult': 'pedestrian', 'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'human.pedestrian.personal_mobility': 'pedestrian',
}


def load_gt_trajectories(nusc, scene_subset=None):
    """{scene_token: {instance_token: [(sample_idx, sample_token, [x,y], tracking_name)]}}"""
    val_names = set(VAL_SCENES) if scene_subset is None else set(scene_subset)
    out = defaultdict(lambda: defaultdict(list))
    for scene in nusc.scene:
        if scene['name'] not in val_names:
            continue
        st = scene['token']
        sample_tok = scene['first_sample_token']
        idx = 0
        while sample_tok:
            sample = nusc.get('sample', sample_tok)
            for ann_tok in sample['anns']:
                ann = nusc.get('sample_annotation', ann_tok)
                tname = CATEGORY_MAP.get(ann['category_name'])
                if tname is None:
                    continue
                inst = ann['instance_token']
                pos = ann['translation'][:2]
                out[st][inst].append((idx, sample_tok, [float(pos[0]), float(pos[1])], tname))
            sample_tok = sample['next']
            idx += 1
    return out


def load_preds(submission, nusc):
    """
    preds_by_class: {scene: {cls: {fidx: [(tid, [x,y])]}}}
    preds_any:      {scene: {fidx: [(tid, cls, [x,y])]}}  (for cross-class diagnostic)
    first_seen:     {scene: {cls: {tid: first_fidx}}}
    """
    sample_to_idx = {}
    for scene in nusc.scene:
        sample_tok = scene['first_sample_token']
        idx = 0
        while sample_tok:
            sample_to_idx[sample_tok] = (scene['token'], idx)
            sample = nusc.get('sample', sample_tok)
            sample_tok = sample['next']
            idx += 1

    preds_by_class = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    preds_any = defaultdict(lambda: defaultdict(list))
    first_seen = defaultdict(lambda: defaultdict(dict))
    for sample_tok, dets in submission['results'].items():
        if sample_tok not in sample_to_idx:
            continue
        st, sidx = sample_to_idx[sample_tok]
        for det in dets:
            tname = det.get('tracking_name')
            if tname not in TRACKING_NAMES:
                continue
            tid = str(det['tracking_id'])
            pos = [float(det['translation'][0]), float(det['translation'][1])]
            preds_by_class[st][tname][sidx].append((tid, pos))
            preds_any[st][sidx].append((tid, tname, pos))
            cur = first_seen[st][tname].get(tid)
            if cur is None or sidx < cur:
                first_seen[st][tname][tid] = sidx
    return preds_by_class, preds_any, first_seen


def per_class_match(gt_instances, preds_cls, n_frames):
    """Per-class Hungarian matching (2m threshold).

    gt_instances: {inst_tok: [(idx, _, pos, tname)]}  — single class
    preds_cls: {fidx: [(tid, pos)]}
    Returns: {inst_tok: [(fidx, pred_id_or_None, dist_or_None)]}
    """
    matched = {inst: [] for inst in gt_instances}
    # For each frame, gather GT entries for this class
    for fidx in range(n_frames):
        gt_at_f = []
        for inst, seq in gt_instances.items():
            for (idx, _, pos, _) in seq:
                if idx == fidx:
                    gt_at_f.append((inst, pos))
                    break
        pred_at_f = preds_cls.get(fidx, [])
        if not gt_at_f:
            continue
        if not pred_at_f:
            for inst, _ in gt_at_f:
                matched[inst].append((fidx, None, None))
            continue
        G, P = len(gt_at_f), len(pred_at_f)
        cost = np.full((G, P), 1e6)
        for gi, (_, gpos) in enumerate(gt_at_f):
            gp = np.array(gpos)
            for pi, (_, ppos) in enumerate(pred_at_f):
                d = float(np.linalg.norm(gp - np.array(ppos)))
                if d <= DIST_TH:
                    cost[gi, pi] = d
        r, c = linear_sum_assignment(cost)
        assigned = {}
        for gi, pi in zip(r, c):
            if cost[gi, pi] < 1e6:
                assigned[gi] = (pi, cost[gi, pi])
        for gi, (inst, _) in enumerate(gt_at_f):
            if gi in assigned:
                pi, d = assigned[gi]
                tid, _ = pred_at_f[pi]
                matched[inst].append((fidx, tid, d))
            else:
                matched[inst].append((fidx, None, None))
    return matched


def classify_events(matched_seq, first_seen_cls):
    """Classify transitions in matched_seq.
    matched_seq: [(fidx, pid_or_None, dist_or_None)]
    first_seen_cls: {tid: first_fidx} for this scene & class
    Returns: list of (frame_prev, frame_new, gap, prev_pid, new_pid, category)
    """
    events = []
    prev = None  # (fidx, pid)
    gap = 0
    for (fidx, pid, _) in matched_seq:
        if pid is None:
            if prev is not None:
                gap += 1
            continue
        if prev is None:
            prev = (fidx, pid); gap = 0
            continue
        if pid != prev[1]:
            if gap >= 2:
                cat = 'track_break_recover'
            elif first_seen_cls.get(pid, -1) == fidx:
                cat = 'newborn_fp'
            else:
                cat = 'same_class_swap'
            events.append((prev[0], fidx, gap, prev[1], pid, cat))
        prev = (fidx, pid); gap = 0
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--limit-scenes', type=int, default=0,
                    help='If >0, only first N val scenes (smoke test)')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.results), f'ids_decompose_{args.label}')
    os.makedirs(out_dir, exist_ok=True)

    log.info('Loading NuScenes API...')
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    val_scene_subset = None
    if args.limit_scenes > 0:
        val_scene_subset = list(VAL_SCENES)[:args.limit_scenes]
        log.info('Limited to first %d val scenes (smoke)', args.limit_scenes)

    log.info('Phase A: load GT trajectories')
    gt_traj = load_gt_trajectories(nusc, val_scene_subset)
    n_inst = sum(len(v) for v in gt_traj.values())
    n_anns = sum(len(seq) for v in gt_traj.values() for seq in v.values())
    log.info('  scenes=%d, GT instances=%d, GT annotations=%d', len(gt_traj), n_inst, n_anns)

    log.info('Phase B: load predictions')
    submission = json.load(open(args.results))
    preds_by_class, preds_any, first_seen = load_preds(submission, nusc)
    n_pred_dets = sum(len(d) for cd in preds_by_class.values() for fd in cd.values() for d in fd.values())
    log.info('  pred dets (in tracking classes)=%d', n_pred_dets)

    log.info('Phase C: per-class Hungarian matching (2m)')
    # Per-instance matched seq: {scene: {inst: [(fidx, pid, dist)]}}
    matched_all = defaultdict(dict)
    inst_to_class = {}  # for grouping
    for st, instances in gt_traj.items():
        if not instances:
            continue
        # frames in scene = max idx + 1 from any instance
        n_frames = max(idx for seq in instances.values() for (idx,_,_,_) in seq) + 1
        # Group instances by class
        insts_by_cls = defaultdict(dict)
        for inst, seq in instances.items():
            cls = seq[0][3]
            insts_by_cls[cls][inst] = seq
            inst_to_class[(st, inst)] = cls
        for cls, cls_insts in insts_by_cls.items():
            preds_cls = preds_by_class.get(st, {}).get(cls, {})
            m = per_class_match(cls_insts, preds_cls, n_frames)
            matched_all[st].update(m)

    # Per-class match-rate sanity
    cls_total = Counter(); cls_match = Counter()
    for st, instances in matched_all.items():
        for inst, seq in instances.items():
            cls = inst_to_class[(st, inst)]
            for (_, pid, _) in seq:
                cls_total[cls] += 1
                if pid is not None:
                    cls_match[cls] += 1
    total = sum(cls_total.values()); matched_ct = sum(cls_match.values())
    log.info('  Match rate: %d / %d = %.1f%%',
             matched_ct, total, 100.0*matched_ct/max(total,1))
    for c in TRACKING_NAMES:
        if cls_total[c]:
            log.info('    %-12s %5d / %5d = %.1f%%',
                     c, cls_match[c], cls_total[c],
                     100.0*cls_match[c]/cls_total[c])

    log.info('Phase D: classify events (3 categories) + secondary diagnostics')
    events = []
    detection_fail = []
    cross_class_misdet = []
    summary = defaultdict(lambda: defaultdict(int))  # cat → cls → count

    for st, instances in matched_all.items():
        for inst, seq in instances.items():
            cls = inst_to_class[(st, inst)]
            n = len(seq)
            n_match = sum(1 for e in seq if e[1] is not None)
            mrate = n_match / max(n, 1)
            if mrate < 0.5:
                # Detection failure (class-internal match rate < 50%).
                # Secondary check: any pred of OTHER class within 2m at any frame
                # this instance was annotated? → detector class confusion.
                cross_match_frames = []
                for (fidx, _, gpos, _) in gt_traj[st][inst]:
                    for (tid, ptname, ppos) in preds_any.get(st, {}).get(fidx, []):
                        if ptname == cls:
                            continue
                        d = float(np.linalg.norm(np.array(gpos) - np.array(ppos)))
                        if d <= DIST_TH:
                            cross_match_frames.append((fidx, tid, ptname, d))
                            break
                df_entry = {
                    'scene': st, 'gt_token': inst, 'gt_class': cls,
                    'n_frames': n, 'n_matched': n_match,
                    'cross_class_match_frames': len(cross_match_frames),
                }
                detection_fail.append(df_entry)
                if cross_match_frames:
                    cross_class_misdet.append({
                        **df_entry,
                        'cross_examples': cross_match_frames[:3],
                    })
                continue
            # Classify events
            ev = classify_events(seq, first_seen[st].get(cls, {}))
            for (fp, fn, gap, ppid, npid, cat) in ev:
                events.append({
                    'scene': st, 'gt_token': inst, 'gt_class': cls,
                    'frame_prev': fp, 'frame_new': fn, 'gap': gap,
                    'prev_pred_id': ppid, 'new_pred_id': npid,
                    'category': cat,
                })
                summary[cat][cls] += 1

    # Output
    events_path = os.path.join(out_dir, 'ids_events.jsonl')
    with open(events_path, 'w') as f:
        for e in events: f.write(json.dumps(e) + '\n')
    detfail_path = os.path.join(out_dir, 'detection_fail.jsonl')
    with open(detfail_path, 'w') as f:
        for d in detection_fail: f.write(json.dumps(d) + '\n')
    cross_path = os.path.join(out_dir, 'cross_class_misdet.jsonl')
    with open(cross_path, 'w') as f:
        for d in cross_class_misdet: f.write(json.dumps(d) + '\n')

    cats = ['track_break_recover', 'newborn_fp', 'same_class_swap']
    log.info('=== IDS EVENT SUMMARY [%s] ===', args.label)
    log.info('Primary (per-class 2m Hungarian, single-threshold):')
    log.info('  Total association events: %d', len(events))
    cat_totals = Counter()
    for c in TRACKING_NAMES:
        if not any(summary[cat][c] for cat in cats):
            continue
        row = f'  {c:<12s}'
        rt = 0
        for cat in cats:
            v = summary[cat][c]
            row += f' {cat[:14]:<14s}={v:<5d}'
            rt += v
            cat_totals[cat] += v
        row += f' total={rt}'
        log.info(row)
    log.info('  ----')
    row = '  TOTAL       '
    for cat in cats:
        row += f' {cat[:14]:<14s}={cat_totals[cat]:<5d}'
    row += f' total={sum(cat_totals.values())}'
    log.info(row)

    log.info('Secondary (separate from primary IDS sum):')
    log.info('  detection_fail trajectories (class-internal match rate < 50%%): %d',
             len(detection_fail))
    log.info('  cross_class_misdetection (subset of above with other-class pred within 2m): %d',
             len(cross_class_misdet))
    df_by_cls = Counter(d['gt_class'] for d in detection_fail)
    cm_by_cls = Counter(d['gt_class'] for d in cross_class_misdet)
    for c in TRACKING_NAMES:
        if df_by_cls[c]:
            log.info('    %-12s detection_fail=%d, cross_class_misdet=%d',
                     c, df_by_cls[c], cm_by_cls[c])
    log.info('Outputs: %s', out_dir)


if __name__ == '__main__':
    main()
