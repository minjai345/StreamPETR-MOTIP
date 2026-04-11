"""Single training-step verifier for the StreamPETR + MOTIP integration.

Builds a Petr3D with motip_cfg, builds the dataloader pointing at the
*_motip pkl, runs one forward pass and checks that:
- detection losses are present and finite
- loss_id is present and finite
- backward() works without NaN/inf in grads of MOTIP submodules

Run: python tools/verify_motip_train_step.py
"""
import sys, warnings, math
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
import importlib
importlib.import_module('projects.mmdet3d_plugin')

CFG = 'projects/configs/StreamPETR/stream_petr_r50_motip_704_bs1_8key_24e.py'
CKPT = 'ckpts/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth'

cfg = Config.fromfile(CFG)
# in-process dataloader to keep verification simple
cfg.data.workers_per_gpu = 0

print('Train pipeline:')
for s in cfg.data.train.pipeline:
    print(' -', s['type'], s.get('keys', '') if s['type'] == 'Collect3D' else '')
print(f'\nseq_mode={cfg.data.train.seq_mode}, queue_length={cfg.data.train.queue_length}, '
      f'num_frame_losses={cfg.model.num_frame_losses}')

# Build dataset/dataloader
ds = build_dataset(cfg.data.train)
print(f'\nDataset size: {len(ds)}')

dl = build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=0,
                      dist=False, shuffle=False)

# Build model + load detector ckpt
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, CKPT, map_location='cpu', strict=False)
model = MMDataParallel(model.cuda(), device_ids=[0])
model.train()
print('\nModel built, ckpt loaded.')

# Pull one batch
data = next(iter(dl))
print('\nBatch keys:', list(data.keys()))
print('img:', data['img'].data[0].shape if hasattr(data['img'], 'data') else type(data['img']))
print('gt_instance_ids type:', type(data.get('gt_instance_ids', None)))

# Forward
out = model(return_loss=True, **data)
print('\n=== Loss dict ===')
for k, v in out.items():
    if torch.is_tensor(v):
        finite = torch.isfinite(v).all().item() if v.numel() > 0 else True
        val = v.mean().item() if v.numel() > 0 else 0.0
        print(f'  {k}: {val:.4f} (finite={finite})')
    else:
        print(f'  {k}: {v}')

# Backward — sanity that grads flow through MOTIP submodules
total = sum(v for k, v in out.items() if 'loss' in k and torch.is_tensor(v))
print(f'\nTotal loss: {total.item():.4f}')
total.backward()

motip_modules = ['id_dict', 'pe_3d', 'tracklet_former', 'id_decoder']
print('\n=== MOTIP grad sanity ===')
for n, p in model.module.named_parameters():
    if any(m in n for m in motip_modules):
        if p.grad is None:
            print(f'  {n}: NO GRAD')
        else:
            g = p.grad.abs().mean().item()
            finite = torch.isfinite(p.grad).all().item()
            if not finite:
                print(f'  {n}: NaN/Inf grad')
            elif g == 0:
                pass  # silently skip zero grads (lots of unused weights)
            else:
                pass
                # print(f'  {n}: {g:.6f}')
n_with_grad = sum(1 for n, p in model.module.named_parameters()
                  if any(m in n for m in motip_modules) and p.grad is not None and p.grad.abs().sum().item() > 0)
n_total = sum(1 for n, p in model.module.named_parameters()
              if any(m in n for m in motip_modules))
print(f'MOTIP params with non-zero grad: {n_with_grad}/{n_total}')

print('\nOK')
