"""Quick profiling: where does tracking eval spend time?"""
import os, sys, time, pickle
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

cfg = Config.fromfile('projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e.py')

import importlib
if hasattr(cfg, 'plugin') and cfg.plugin:
    _module_dir = os.path.dirname(cfg.plugin_dir).split('/')
    importlib.import_module('.'.join(_module_dir))

model = build_model(cfg.model)
wrap_fp16_model(model)
model.cuda().eval()
load_checkpoint(model, 'work_dirs/motip_phase1_v1/iter_14064.pth',
                map_location='cuda', strict=False)

val_cfg = cfg.data.val.copy()
val_cfg['test_mode'] = True
val_dataset = build_dataset(val_cfg)

from projects.mmdet3d_plugin.models.utils.misc import locations

print(f"Dataset size: {len(val_dataset)}")
print(f"Profiling 20 frames...")

times = {'data_load': [], 'to_gpu': [], 'feat_extract': [], 'head_forward': []}

model.prev_scene_token = None

for idx in range(20):
    # Data loading
    t0 = time.time()
    data = val_dataset[idx]
    t1 = time.time()
    times['data_load'].append(t1 - t0)

    # To GPU
    img = data['img'][0].data.unsqueeze(0).cuda()
    img_metas = [data['img_metas'][0].data]
    data_dict = {}
    for k in ['lidar2img', 'intrinsics', 'extrinsics',
              'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']:
        if k in data:
            val = data[k][0] if isinstance(data[k], list) else data[k]
            if hasattr(val, 'data'):
                val = val.data
            if isinstance(val, torch.Tensor):
                data_dict[k] = val.unsqueeze(0).cuda()
    data_dict['img'] = img
    t2 = time.time()
    times['to_gpu'].append(t2 - t1)

    # Feature extraction
    with torch.no_grad():
        data_dict['img_feats'] = model.extract_img_feat(img, 1)
    torch.cuda.synchronize()
    t3 = time.time()
    times['feat_extract'].append(t3 - t2)

    # Head forward
    scene_tok = img_metas[0].get('scene_token', '')
    if scene_tok != model.prev_scene_token:
        model.prev_scene_token = scene_tok
        data_dict['prev_exists'] = img.new_zeros(1)
        model.pts_bbox_head.reset_memory()
    else:
        data_dict['prev_exists'] = img.new_ones(1)

    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    bs, n = data_dict['img_feats'].shape[:2]
    x = data_dict['img_feats'].flatten(0, 1)
    location = locations(x, model.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)

    with torch.no_grad():
        outs = model.pts_bbox_head(location, img_metas, None, **data_dict)
    torch.cuda.synchronize()
    t4 = time.time()
    times['head_forward'].append(t4 - t3)

    print(f"[{idx:2d}] load={t1-t0:.3f}s  gpu={t2-t1:.3f}s  feat={t3-t2:.3f}s  head={t4-t3:.3f}s")

print("\n=== Average (skip first 2 warmup) ===")
for k, v in times.items():
    avg = sum(v[2:]) / len(v[2:])
    print(f"  {k:15s}: {avg:.3f}s")
total = sum(sum(v[2:]) / len(v[2:]) for v in times.values())
print(f"  {'TOTAL':15s}: {total:.3f}s")
print(f"\nEstimated full eval: {total * 6019:.0f}s = {total * 6019 / 60:.1f}min")
