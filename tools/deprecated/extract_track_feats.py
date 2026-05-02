"""[DEPRECATED — DO NOT USE]

Output format is incompatible with tools/eval_tracking.py:
  - This script: dict[token] of pre-selected top-300 features
  - eval_tracking.py: list[idx] of {'pts_bbox': {raw 428 features}}

For tracking eval, ALWAYS use the standard mmdet pipeline:
    bash tools/dist_test.sh <CONFIG> <CKPT> <NUM_GPUS> --out track_feats.pkl
    python tools/eval_tracking.py --config <CONFIG> --checkpoint <CKPT> \
        --feats track_feats.pkl --det-thresh 0.25 --new-thresh 0.40 --id-thresh 0.10

Kept for reference only.
"""
import os, sys, time, argparse, pickle
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--workers', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        _module_dir = os.path.dirname(cfg.plugin_dir).split('/')
        importlib.import_module('.'.join(_module_dir))

    model = build_model(cfg.model)
    wrap_fp16_model(model)
    model.cuda().eval()
    print('Loading checkpoint:', args.checkpoint)
    load_checkpoint(model, args.checkpoint, map_location='cuda', strict=False)
    model = MMDataParallel(model, device_ids=[0])

    val_cfg = cfg.data.val.copy()
    val_cfg['test_mode'] = True
    dataset = build_dataset(val_cfg)
    print(f'Dataset: {len(dataset)} samples')

    from torch.utils.data import DataLoader
    from functools import partial
    from mmcv.parallel import collate
    from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=1),
        pin_memory=True,
    )

    results = {}
    inner = model.module
    t_start = time.time()

    for i, data in enumerate(loader):
        with torch.no_grad():
            # Call forward_test exactly as mmdet3d does
            # forward_test unwraps DataContainers internally
            result = model(return_loss=False, rescale=True, **data)

            # Read cached outs from simple_test_pts
            outs = inner._test_outs
            cls_scores = outs['all_cls_scores'][-1][0]
            bbox_preds = outs['all_bbox_preds'][-1][0]
            query_feat = outs['query_feat'][0]

            cls_sig = cls_scores.sigmoid()
            scores_flat, indexs = cls_sig.view(-1).topk(300)
            num_classes = cls_scores.shape[1]
            labels = indexs % num_classes
            query_indices = torch.div(indexs, num_classes, rounding_mode='trunc')
            decoded_bbox = denormalize_bbox(bbox_preds[query_indices], pc_range)

        info = dataset.data_infos[i]
        token = info['token']
        scene_token = info.get('scene_token', '')

        # ego_pose: build from info
        from pyquaternion import Quaternion
        import numpy as np
        l2e_r = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_t = np.array(info['lidar2ego_translation'])
        e2g_r = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_t = np.array(info['ego2global_translation'])
        l2g = np.eye(4)
        l2g[:3, :3] = e2g_r @ l2e_r
        l2g[:3, 3] = e2g_r @ l2e_t + e2g_t

        results[token] = {
            'scores': scores_flat.cpu(),
            'labels': labels.cpu(),
            'bbox_decoded': decoded_bbox.cpu(),
            'bbox_raw': bbox_preds[query_indices].cpu(),
            'query_feat': query_feat[query_indices].cpu(),
            'ego_pose': torch.from_numpy(l2g).float(),
            'scene_token': scene_token,
        }

        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            eta = (len(dataset) - i - 1) / fps
            print(f'[{i+1}/{len(dataset)}] {fps:.1f} fps, ETA {eta/60:.1f}min')

    elapsed = time.time() - t_start
    print(f'Done: {len(dataset)} samples in {elapsed:.1f}s ({len(dataset)/elapsed:.1f} fps)')

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), 'track_feats.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    fsize = os.path.getsize(out_path) / 1024**3
    print(f'Saved: {out_path} ({fsize:.2f} GB)')


if __name__ == '__main__':
    main()
