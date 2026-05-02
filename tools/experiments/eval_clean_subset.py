"""Re-run nuScenes TrackingEval on the subset of val scenes that did NOT
hit pool_full in gt_aligned_eval v2 (clean signal — no K=50 contamination).

Usage:
    python tools/experiments/eval_clean_subset.py \\
        --counts work_dirs/.../gt_aligned_v2/gt_aligned_counts.json \\
        --results work_dirs/.../tracking_results_*.json \\
        --label baseline_or_oracle_or_v2 \\
        --dataroot data/nuscenes
"""
import os, sys, json, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

import numpy as np
from nuscenes import NuScenes
import nuscenes.utils.splits as nusc_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--counts', required=True,
                    help='gt_aligned_counts.json with per_scene token→counts (defines clean subset)')
    ap.add_argument('--results', required=True, help='Path to tracking_results JSON')
    ap.add_argument('--label', required=True, help='Label for this run (used in out paths/log)')
    ap.add_argument('--dataroot', default='data/nuscenes')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--limit-scenes', type=int, default=0,
                    help='If >0, only use first N clean scenes (for smoke test)')
    args = ap.parse_args()

    counts = json.load(open(args.counts))
    clean_tokens = [st for st, c in counts['per_scene'].items()
                    if c['matched_pool_full'] == 0]
    log.info('Clean scenes (no pool_full): %d', len(clean_tokens))
    if args.limit_scenes > 0:
        clean_tokens = clean_tokens[:args.limit_scenes]
        log.info('Limited to first %d clean scenes', len(clean_tokens))

    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=False)
    clean_token_set = set(clean_tokens)
    clean_names = []
    for st in clean_tokens:
        sc = nusc.get('scene', st)
        clean_names.append(sc['name'])
    clean_names_set = set(clean_names)
    log.info('Mapped to %d scene names', len(clean_names))

    # Collect sample_tokens belonging to clean scenes
    clean_sample_tokens = set()
    for st in clean_tokens:
        sc = nusc.get('scene', st)
        st_sample = sc['first_sample_token']
        while st_sample:
            clean_sample_tokens.add(st_sample)
            sample = nusc.get('sample', st_sample)
            st_sample = sample['next']
    log.info('Clean sample tokens: %d', len(clean_sample_tokens))

    # Filter results JSON to clean samples only
    submission = json.load(open(args.results))
    filtered = {tok: dets for tok, dets in submission['results'].items()
                if tok in clean_sample_tokens}
    # Add empty entries for any clean-scene samples missing from submission
    for tok in clean_sample_tokens:
        if tok not in filtered:
            filtered[tok] = []
    log.info('Filtered submission: %d / %d sample entries kept (clean tokens covered: %d)',
             len(filtered), len(submission['results']), len(clean_sample_tokens))

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.results), f'clean91_{args.label}')
    os.makedirs(out_dir, exist_ok=True)
    filtered_path = os.path.join(out_dir, 'tracking_results_clean91.json')
    with open(filtered_path, 'w') as f:
        json.dump({'meta': submission['meta'], 'results': filtered}, f)
    log.info('Saved filtered submission: %s', filtered_path)

    # Monkey-patch only the returned dict, NOT splits.val itself.
    # (Original create_splits_scenes asserts len(all_scenes)==1000, so
    # mutating splits.val would break it.)
    orig_create = nusc_splits.create_splits_scenes
    def patched_create(*a, **kw):
        s = orig_create(*a, **kw)
        s['val'] = list(clean_names)
        return s
    nusc_splits.create_splits_scenes = patched_create
    # Patch into already-imported modules (loaders.py imports the symbol)
    import sys as _sys
    for modname, mod in list(_sys.modules.items()):
        if modname.startswith('nuscenes') and getattr(mod, 'create_splits_scenes', None) is orig_create:
            mod.create_splits_scenes = patched_create

    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory
    eval_cfg = config_factory('tracking_nips_2019')
    te = TrackingEval(config=eval_cfg, result_path=filtered_path, eval_set='val',
                      output_dir=out_dir, nusc_version='v1.0-trainval',
                      nusc_dataroot=args.dataroot, verbose=False)
    res = te.main()
    summ = res if isinstance(res, dict) else res.serialize()
    lm = summ.get('label_metrics', {})
    amota = np.nanmean(list(lm['amota'].values())) if 'amota' in lm else 0
    mota = np.nanmean(list(lm['mota'].values())) if 'mota' in lm else 0
    ids = int(np.nansum(list(lm['ids'].values()))) if 'ids' in lm else 0
    log.info('=== CLEAN-91 SUBSET [%s] ===', args.label)
    log.info(f'AMOTA={amota:.4f} MOTA={mota:.4f} IDS={ids}')


if __name__ == '__main__':
    main()
