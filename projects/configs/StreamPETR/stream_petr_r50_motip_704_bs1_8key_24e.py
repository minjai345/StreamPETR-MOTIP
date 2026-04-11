"""StreamPETR + MOTIP, frozen detector (Phase 1).

Inherits the 428q non-flash config (which inherits the 60e R50 NuImg config),
then overrides the dataset / sampler to switch from seq_mode (1 frame per
sample) to sliding-window training (8 frames per sample). This matches the
training procedure described in the paper and is the natural fit for MOTIP's
clip-based id loss.

Differences vs the base 60e config:
- queue_length: 1 -> 8
- num_frame_losses: 1 -> 2
- num_frame_head_grads / num_frame_backbone_grads: 1 -> 2
- seq_mode: True -> False
- seq_split_num: 2 -> 1
- shuffler_sampler: InfiniteGroupEachSampleInBatchSampler -> DistributedGroupSampler
- pipeline: replace ObjectRangeFilter / ObjectNameFilter with WithIDs variants,
  add WrapInstanceIDs, and add gt_instance_ids to Collect3D keys.
- model: add motip_cfg (frozen detector for Phase 1)
- optimizer: lr_mult=0 on detector params so the optimizer literally cannot
  move them, in addition to the in-graph .detach() inside compute_id_loss.
- load_from: pretrained 60e ckpt
"""
_base_ = ['./stream_petr_r50_noflash_704_bs2_seq_428q_nui_60e.py']

# ── MOTIP module config ──
motip_cfg = dict(
    num_ids=50,
    embed_dim=256,
    id_decoder_layers=6,
    id_decoder_heads=8,
    id_decoder_dropout=0.1,
    id_loss_weight=1.0,
    freeze_detector=True,   # Phase 1
    context_len=5,
)

# Sliding-window training: each sample is 8 consecutive frames; only the last
# 2 carry detection gradient/loss (matches the paper's bs1_8key_2grad setup).
queue_length = 8
num_frame_losses = 2

model = dict(
    motip_cfg=motip_cfg,
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
)

# Pipeline: use the ID-aware filters and add gt_instance_ids to Collect3D.
# We rebuild the train pipeline from scratch (rather than mutating _base_'s
# list in place) so the override is explicit and easy to read.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = {
    "resize_lim": (0.38, 0.55),
    "final_dim": (256, 704),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                'img_timestamp', 'ego_pose', 'ego_pose_inv']

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilterWithIDs', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilterWithIDs', classes=class_names),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    dict(type='GlobalRotScaleTransImage',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='WrapInstanceIDs'),
    dict(type='PETRFormatBundle3D', class_names=class_names,
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D',
         keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
               'centers2d', 'depths', 'prev_exists', 'gt_instance_ids'] + collect_keys,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d')),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_train_motip.pkl',
        seq_mode=False,           # disable streaming, switch to sliding window
        seq_split_num=1,
        num_frame_losses=num_frame_losses,
        queue_length=queue_length,
        pipeline=train_pipeline,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas', 'gt_instance_ids'],
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
)

# ── Optimizer: freeze detector via lr_mult=0, train MOTIP submodules only ──
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone':  dict(lr_mult=0.0),
            'img_neck':      dict(lr_mult=0.0),
            'img_roi_head':  dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            'id_dict':         dict(lr_mult=1.0),
            'pe_3d':           dict(lr_mult=1.0),
            'tracklet_former': dict(lr_mult=1.0),
            'id_decoder':      dict(lr_mult=1.0),
        }),
    weight_decay=0.01)

# Load pretrained detector weights
load_from = 'ckpts/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth'

# Phase 1 is short — id loss converges quickly relative to detection
total_epochs = 12
runner = dict(type='IterBasedRunner', max_iters=12 * (28130 // 1))  # placeholder; mmcv recomputes
