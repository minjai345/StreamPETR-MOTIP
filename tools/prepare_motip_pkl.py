"""Add instance_tokens to StreamPETR's nuscenes pkl files for MOTIP.

StreamPETR's `nuscenes2d_temporal_infos_*.pkl` does not include instance_tokens
(per-detection unique IDs), but MOTIP requires them for ID supervision.

We borrow them from SparseBEV's `nuscenes_infos_*_sweep.pkl`, which has the
exact same gt_boxes (verified) and contains `instance_tokens`.

Output: writes new pkl files alongside the originals with `_motip` suffix.
Original files are not modified.
"""
import os
import pickle
import argparse
import numpy as np


def merge_one(sb_pkl_path, sp_pkl_path, out_pkl_path):
    print(f'\n[merge] {os.path.basename(sp_pkl_path)} <- {os.path.basename(sb_pkl_path)}')
    with open(sb_pkl_path, 'rb') as f:
        sb = pickle.load(f)
    with open(sp_pkl_path, 'rb') as f:
        sp = pickle.load(f)

    sb_by_token = {info['token']: info for info in sb['infos']}

    n_total = len(sp['infos'])
    n_added = 0
    n_skipped = 0
    n_box_mismatch = 0

    for sp_info in sp['infos']:
        sb_info = sb_by_token.get(sp_info['token'])
        if sb_info is None:
            n_skipped += 1
            sp_info['instance_tokens'] = []
            continue

        sb_box = sb_info['gt_boxes']
        sp_box = sp_info['gt_boxes']
        if sb_box.shape != sp_box.shape or not np.allclose(sb_box, sp_box, atol=1e-3):
            n_box_mismatch += 1
            sp_info['instance_tokens'] = []
            continue

        sp_info['instance_tokens'] = list(sb_info['instance_tokens'])
        n_added += 1

    print(f'  total: {n_total}, added: {n_added}, skipped: {n_skipped}, box_mismatch: {n_box_mismatch}')

    with open(out_pkl_path, 'wb') as f:
        pickle.dump(sp, f)
    print(f'  written: {out_pkl_path}')

    # Sanity check
    with open(out_pkl_path, 'rb') as f:
        check = pickle.load(f)
    s0 = check['infos'][0]
    assert 'instance_tokens' in s0, 'instance_tokens missing in output'
    if n_added > 0:
        assert len(s0['instance_tokens']) == len(s0['gt_boxes']), \
            f'length mismatch in sample 0: {len(s0["instance_tokens"])} vs {len(s0["gt_boxes"])}'
    print('  sanity check passed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sb-root', default='/data4/minjae/workspace/SparseBEV/data/nuscenes')
    parser.add_argument('--sp-root', default='/data4/minjae/workspace/StreamPETR/data/nuscenes')
    args = parser.parse_args()

    pairs = [
        ('nuscenes_infos_train_sweep.pkl', 'nuscenes2d_temporal_infos_train.pkl', 'nuscenes2d_temporal_infos_train_motip.pkl'),
        ('nuscenes_infos_val_sweep.pkl',   'nuscenes2d_temporal_infos_val.pkl',   'nuscenes2d_temporal_infos_val_motip.pkl'),
    ]

    for sb_name, sp_name, out_name in pairs:
        sb_path = os.path.join(args.sb_root, sb_name)
        sp_path = os.path.join(args.sp_root, sp_name)
        out_path = os.path.join(args.sp_root, out_name)
        if not os.path.isfile(sb_path):
            print(f'[skip] {sb_path} not found')
            continue
        if not os.path.isfile(sp_path):
            print(f'[skip] {sp_path} not found')
            continue
        merge_one(sb_path, sp_path, out_path)


if __name__ == '__main__':
    main()
