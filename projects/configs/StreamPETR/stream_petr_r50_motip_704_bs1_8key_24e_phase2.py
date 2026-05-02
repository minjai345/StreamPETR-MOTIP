"""E2E joint training (PF-Track style).

Loads StreamPETR 60e detector pretrain. MOTIP modules start from scratch.
Joint training with detection loss + MOTIP id loss.
Fresh run (no Phase 1 resume): single GPU, bs=6, 6 epochs.
"""
_base_ = ['./stream_petr_r50_motip_704_bs1_8key_24e.py']

# Phase 2: detector unfreeze
motip_cfg = dict(
    num_ids=50,
    embed_dim=256,
    id_decoder_layers=6,
    id_decoder_heads=8,
    id_decoder_dropout=0.1,
    id_loss_weight=2.0,  # boost vs ~30 detection loss terms
    freeze_detector=False,  # ← unfreeze
    context_len=5,
)

num_frame_losses = 1  # base MOTIP set 2, but E2E + with_cp=False OOMs on 24GB

model = dict(
    motip_cfg=motip_cfg,
    num_frame_losses=num_frame_losses,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    img_backbone=dict(with_cp=True),
    pts_bbox_head=dict(
        num_frame_losses=num_frame_losses,
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(with_cp=True))))
)

find_unused_parameters = True

# Optimizer: detector parts get small lr_mult (fine-tune), MOTIP stays at 1.0
# MOTIP Phase 1 already converged, so MOTIP lr can be smaller too.
optimizer = dict(
    type='AdamW',
    lr=1.5e-4,  # Phase 1 was 2e-4 at bs=8; bs=6 → linear scale to 1.5e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone':  dict(lr_mult=0.1),   # fine-tune backbone
            'img_neck':      dict(lr_mult=0.1),
            'img_roi_head':  dict(lr_mult=0.1),
            'pts_bbox_head': dict(lr_mult=0.1),
            'id_dict':         dict(lr_mult=1.0),
            'pe_3d':           dict(lr_mult=1.0),
            'tracklet_former': dict(lr_mult=1.0),
            'id_decoder':      dict(lr_mult=1.0),
        }),
    weight_decay=0.01)

# E2E schedule: single GPU, samples_per_gpu=6 (with_cp=True saves VRAM)
# → effective batch=6, 1 epoch = 28130/6 = 4688 iter
# NOTE: base MOTIP config sets data.samples_per_gpu=4 — must override explicitly.
num_gpus = 1
batch_size = 6
data = dict(samples_per_gpu=batch_size, workers_per_gpu=4)
num_iters_per_epoch = 28130 // (num_gpus * batch_size)  # 4688
num_epochs = 6
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
evaluation = dict(interval=num_iters_per_epoch * num_epochs)

# LR schedule: shorter warmup + CosineAnnealing
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,  # shorter than Phase 1 (500)
    warmup_ratio=1/3,
    min_lr_ratio=1e-3)

# Override wandb run name so it doesn't look like Phase 1
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='StreamPETR-MOTIP',
                 name='e2e_v2_fresh_bs6_6ep',
                 config=dict(
                     phase='e2e_joint',
                     architecture='r50_428q_nui',
                     queue_length=8,
                     num_frame_losses=num_frame_losses,
                     num_epochs=num_epochs,
                     num_gpus=num_gpus,
                     samples_per_gpu=batch_size,
                     freeze_detector=False,
                     id_loss_weight=2.0,
                     detector_lr_mult=0.1,
                     with_cp=True,
                 ))),
    ])
