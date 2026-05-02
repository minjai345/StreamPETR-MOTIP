"""Slot Flip Rate (실험 B) — measure baseline tracker's per-GT-instance
slot consistency rate. Reuses ids_event_decompose's GT/pred loaders + per-class
Hungarian matching (2m).

For each GT instance trajectory, compute:
  - adj_strict = P(pid_t == pid_{t-1} | matched at both t and t-1)
  - persistent = P(pid_t == pid_first | matched at t, t > first)
  - with_drop  = P(pid_t == pid_{t-1} | matched at t-1)
                 (unmatched at t counts as flip — fair denom vs v2 matched_correct)

Compare to v2 matched_correct rate (78.8% on clean91).

Usage:
    python tools/experiments/slot_flip_rate.py \\
        --results <baseline.json> --label baseline_consistency \\
        --counts work_dirs/.../gt_aligned_v2/gt_aligned_counts.json \\
        --out-dir work_dirs/.../slot_flip_baseline/
"""
import os, sys, json, argparse, logging
from collections import defaultdict, Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.splits import val as VAL_SCENES

from tools.experiments.ids_event_decompose import (
    load_gt_trajectories, load_preds, per_class_match,
    TRACKING_NAMES, DIST_TH,
)
from scipy.optimize import linear_sum_assignment


def class_agnostic_match(gt_instances_all, preds_any, n_frames):
    """Class-AGNOSTIC greedy 1-to-1 matching (mirrors v2 tracker's internal
    matching: any pred to any GT, greedy by min distance, 2m threshold).

    gt_instances_all: {inst_tok: [(idx, _, pos, tname)]}  (mixed classes)
    preds_any: {fidx: [(tid, tname, pos)]}
    Returns: {inst_tok: [(fidx, pred_id_or_None, dist_or_None)]}
    """
    matched = {inst: [] for inst in gt_instances_all}
    for fidx in range(n_frames):
        gt_at_f = []
        for inst, seq in gt_instances_all.items():
            for (idx, _, pos, _) in seq:
                if idx == fidx:
                    gt_at_f.append((inst, pos))
                    break
        pred_at_f = preds_any.get(fidx, [])
        if not gt_at_f:
            continue
        if not pred_at_f:
            for inst, _ in gt_at_f:
                matched[inst].append((fidx, None, None))
            continue
        # Greedy by min distance (same as v2 tracker)
        cand = []
        for gi, (_, gpos) in enumerate(gt_at_f):
            gp = np.array(gpos)
            for pi, (tid, _, ppos) in enumerate(pred_at_f):
                d = float(np.linalg.norm(gp - np.array(ppos)))
                if d <= DIST_TH:
                    cand.append((d, gi, pi))
        cand.sort()
        taken_g, taken_p = set(), set()
        assigned = {}
        for d, gi, pi in cand:
            if gi in taken_g or pi in taken_p:
                continue
            taken_g.add(gi); taken_p.add(pi)
            assigned[gi] = (pi, d)
        for gi, (inst, _) in enumerate(gt_at_f):
            if gi in assigned:
                pi, d = assigned[gi]
                tid, _, _ = pred_at_f[pi]
                matched[inst].append((fidx, tid, d))
            else:
                matched[inst].append((fidx, None, None))
    return matched


def compute_consistency(matched_seq):
    """Return dict of (n_adj_pairs, n_adj_consistent, n_persistent_pairs, n_persistent,
    n_drop_pairs, n_drop_consistent)."""
    # Filter to matched-only entries with frame indices preserved
    n = len(matched_seq)
    if n == 0:
        return None

    n_adj_pairs = n_adj_consistent = 0
    n_pers_pairs = n_pers_consistent = 0
    n_drop_pairs = n_drop_consistent = 0

    first_pid = None
    prev_pid = None
    prev_matched = False  # was the previous frame entry matched?
    for (fidx, pid, _) in matched_seq:
        if first_pid is None and pid is not None:
            first_pid = pid
        if prev_matched:
            n_drop_pairs += 1
            if pid is not None and pid == prev_pid:
                n_drop_consistent += 1
            if pid is not None:
                n_adj_pairs += 1
                if pid == prev_pid:
                    n_adj_consistent += 1
        if pid is not None and first_pid is not None and pid != first_pid:
            n_pers_pairs += 1
            if pid == first_pid:
                n_pers_consistent += 1
        elif pid is not None and first_pid is not None and pid == first_pid:
            n_pers_pairs += 1
            n_pers_consistent += 1
        # update state
        prev_pid = pid
        prev_matched = pid is not None
    # Persistent excludes first frame
    if n_pers_pairs > 0:
        n_pers_pairs -= 1  # exclude the first matched frame (no prev to compare against)
        n_pers_consistent -= 1  # the first frame trivially matches itself; remove
    return {
        'n_adj_pairs': n_adj_pairs, 'n_adj_consistent': n_adj_consistent,
        'n_pers_pairs': n_pers_pairs, 'n_pers_consistent': n_pers_consistent,
        'n_drop_pairs': n_drop_pairs, 'n_drop_consistent': n_drop_consistent,
    }


def aggregate_consistency(matched_all, inst_to_class, scene_filter=None):
    """Aggregate per-class and overall."""
    per_class = defaultdict(lambda: {'adj_p': 0, 'adj_c': 0, 'pers_p': 0, 'pers_c': 0,
                                     'drop_p': 0, 'drop_c': 0, 'instances': 0})
    for st, instances in matched_all.items():
        if scene_filter is not None and st not in scene_filter:
            continue
        for inst, seq in instances.items():
            cls = inst_to_class[(st, inst)]
            r = compute_consistency(seq)
            if r is None: continue
            per_class[cls]['adj_p']  += r['n_adj_pairs']
            per_class[cls]['adj_c']  += r['n_adj_consistent']
            per_class[cls]['pers_p'] += r['n_pers_pairs']
            per_class[cls]['pers_c'] += r['n_pers_consistent']
            per_class[cls]['drop_p'] += r['n_drop_pairs']
            per_class[cls]['drop_c'] += r['n_drop_consistent']
            per_class[cls]['instances'] += 1
    return per_class


def report(per_class, label):
    log.info('=== %s ===', label)
    log.info('%-12s %-9s %-12s %-12s %-12s', 'class', 'inst', 'adj_strict', 'persistent', 'with_drop')
    log.info('-' * 60)
    tot = {'adj_p': 0, 'adj_c': 0, 'pers_p': 0, 'pers_c': 0, 'drop_p': 0, 'drop_c': 0, 'instances': 0}
    for c in TRACKING_NAMES:
        if c not in per_class: continue
        d = per_class[c]
        for k in tot: tot[k] += d[k]
        adj = 100.0 * d['adj_c'] / max(d['adj_p'], 1)
        pers = 100.0 * d['pers_c'] / max(d['pers_p'], 1)
        drop = 100.0 * d['drop_c'] / max(d['drop_p'], 1)
        log.info('%-12s %-9d %-12s %-12s %-12s', c, d['instances'],
                 f'{adj:.1f}% ({d["adj_c"]}/{d["adj_p"]})',
                 f'{pers:.1f}% ({d["pers_c"]}/{d["pers_p"]})',
                 f'{drop:.1f}% ({d["drop_c"]}/{d["drop_p"]})')
    log.info('-' * 60)
    adj = 100.0 * tot['adj_c'] / max(tot['adj_p'], 1)
    pers = 100.0 * tot['pers_c'] / max(tot['pers_p'], 1)
    drop = 100.0 * tot['drop_c'] / max(tot['drop_p'], 1)
    log.info('%-12s %-9d %-12s %-12s %-12s', 'TOTAL', tot['instances'],
             f'{adj:.1f}% ({tot["adj_c"]}/{tot["adj_p"]})',
             f'{pers:.1f}% ({tot["pers_c"]}/{tot["pers_p"]})',
             f'{drop:.1f}% ({tot["drop_c"]}/{tot["drop_p"]})')
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--counts', default=None,
                    help='gt_aligned_counts.json — for clean91 subset comparison')
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--limit-scenes', type=int, default=0)
    ap.add_argument('--matching', choices=['class-aware', 'class-agnostic'],
                    default='class-aware',
                    help='class-aware: per-class Hungarian (matches nuScenes IDS);'
                         ' class-agnostic: greedy any-class (matches v2 tracker internal)')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.results), f'slot_flip_{args.label}')
    os.makedirs(out_dir, exist_ok=True)

    log.info('Loading NuScenes API...')
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)

    val_scene_subset = None
    if args.limit_scenes > 0:
        val_scene_subset = list(VAL_SCENES)[:args.limit_scenes]
        log.info('Limited to first %d val scenes (smoke)', args.limit_scenes)

    log.info('Phase A: load GT trajectories')
    gt_traj = load_gt_trajectories(nusc, val_scene_subset)
    log.info('  scenes=%d, GT instances=%d',
             len(gt_traj), sum(len(v) for v in gt_traj.values()))

    log.info('Phase B: load predictions')
    submission = json.load(open(args.results))
    preds_by_class, preds_any, first_seen = load_preds(submission, nusc)

    log.info('Phase C: matching (%s, 2m)', args.matching)
    matched_all = defaultdict(dict)
    inst_to_class = {}
    for st, instances in gt_traj.items():
        if not instances: continue
        n_frames = max(idx for seq in instances.values() for (idx,_,_,_) in seq) + 1
        # Always need inst_to_class for reporting
        for inst, seq in instances.items():
            inst_to_class[(st, inst)] = seq[0][3]
        if args.matching == 'class-aware':
            insts_by_cls = defaultdict(dict)
            for inst, seq in instances.items():
                cls = seq[0][3]
                insts_by_cls[cls][inst] = seq
            for cls, cls_insts in insts_by_cls.items():
                preds_cls = preds_by_class.get(st, {}).get(cls, {})
                m = per_class_match(cls_insts, preds_cls, n_frames)
                matched_all[st].update(m)
        else:  # class-agnostic
            m = class_agnostic_match(instances, preds_any.get(st, {}), n_frames)
            matched_all[st].update(m)

    log.info('Phase D: compute consistency rates')

    # Full
    per_class_full = aggregate_consistency(matched_all, inst_to_class)
    tot_full = report(per_class_full, f'{args.label} — FULL ({len(matched_all)} scenes)')

    # Clean91 if counts provided
    if args.counts:
        counts = json.load(open(args.counts))
        clean_scenes = set(st for st, c in counts['per_scene'].items()
                           if c['matched_pool_full'] == 0)
        log.info('Clean scenes from counts: %d', len(clean_scenes))
        per_class_clean = aggregate_consistency(matched_all, inst_to_class,
                                                scene_filter=clean_scenes)
        tot_clean = report(per_class_clean, f'{args.label} — CLEAN91 (%d scenes)' % len(clean_scenes))
    else:
        per_class_clean = None
        tot_clean = None

    # Dump JSON summary
    out_json = os.path.join(out_dir, 'consistency_summary.json')
    summary = {
        'label': args.label,
        'full': {c: dict(per_class_full[c]) for c in per_class_full},
        'full_total': tot_full,
    }
    if per_class_clean is not None:
        summary['clean91'] = {c: dict(per_class_clean[c]) for c in per_class_clean}
        summary['clean91_total'] = tot_clean
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info('Saved summary: %s', out_json)


if __name__ == '__main__':
    main()
