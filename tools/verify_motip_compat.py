"""
StreamPETR forward pass verification + MOTIP compatibility inspection.

Goals:
1. Run a single forward pass with the pretrained checkpoint.
2. Inspect what comes out of the detection head.
3. Identify what we need to expose for MOTIP integration (query_feat).
4. Confirm Petr3D + StreamPETRHead actually runs end-to-end.

Run: python tools/verify_motip_compat.py
"""
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import importlib
importlib.import_module('projects.mmdet3d_plugin')

CFG = 'projects/configs/StreamPETR/stream_petr_r50_noflash_704_bs2_seq_428q_nui_60e.py'
CKPT = 'ckpts/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth'

cfg = Config.fromfile(CFG)
cfg.model.train_cfg = None

# Build dataset (val)
dataset = build_dataset(cfg.data.test)
print(f'Val dataset size: {len(dataset)}')

dataloader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=0,  # main process to avoid worker overhead
    dist=False,
    shuffle=False,
)

# Build model + load checkpoint
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu')
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

# ── Hook the head to capture outs_dec ──
captured = {}
orig_forward = model.module.pts_bbox_head.forward
def hooked_forward(*args, **kwargs):
    out = orig_forward(*args, **kwargs)
    # outs_dec is internal — we read what comes out and check
    print('\n=== Head output dict keys ===')
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}')
        elif v is None:
            print(f'  {k}: None')
        else:
            print(f'  {k}: {type(v).__name__}')
    return out
model.module.pts_bbox_head.forward = hooked_forward

# Forward 1 sample
print('\n=== Running forward on 1 sample ===')
with torch.no_grad():
    for i, data in enumerate(dataloader):
        print(f'Sample {i}: img_metas keys = {list(data["img_metas"][0].data[0][0].keys())[:8]}')
        result = model(return_loss=False, rescale=True, **data)
        print(f'Result type: {type(result).__name__}, len: {len(result)}')
        if isinstance(result, list) and len(result) > 0:
            print(f'Result[0] keys: {list(result[0].keys()) if isinstance(result[0], dict) else type(result[0])}')
            r0 = result[0]
            if isinstance(r0, dict) and 'pts_bbox' in r0:
                pb = r0['pts_bbox']
                print(f'pts_bbox keys: {list(pb.keys())}')
                if 'boxes_3d' in pb:
                    print(f'  boxes_3d: {len(pb["boxes_3d"])} detections')
                if 'scores_3d' in pb:
                    print(f'  scores_3d shape: {pb["scores_3d"].shape}')
                    print(f'  top-5 scores: {pb["scores_3d"][:5].tolist()}')
        break

print('\n=== Done ===')
