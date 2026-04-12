# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations
from projects.mmdet3d_plugin.models.motip import (
    IDDictionary, IDDecoder, Positional3DEncoding, TrackletFormer,
)

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 motip_cfg=None):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False

        # ── MOTIP modules (optional). When motip_cfg is None the detector
        #    behaves exactly like the upstream Petr3D.
        self.motip_cfg = motip_cfg
        if motip_cfg is not None:
            num_ids = motip_cfg.get('num_ids', 50)
            embed_dim = motip_cfg.get('embed_dim', 256)
            self.id_dict = IDDictionary(num_ids=num_ids, embed_dim=embed_dim)
            self.pe_3d = Positional3DEncoding(embed_dim=embed_dim, input_dim=9)
            self.tracklet_former = TrackletFormer(embed_dim=embed_dim)
            self.id_decoder = IDDecoder(
                d_model=3 * embed_dim,
                nhead=motip_cfg.get('id_decoder_heads', 8),
                num_layers=motip_cfg.get('id_decoder_layers', 6),
                num_ids=num_ids,
                dropout=motip_cfg.get('id_decoder_dropout', 0.1),
            )
            self.id_loss_weight = motip_cfg.get('id_loss_weight', 1.0)
            self.freeze_detector = motip_cfg.get('freeze_detector', True)
            self.context_len = motip_cfg.get('context_len', 5)
            self._tracklet_buffers = {}  # per-batch-slot history buffer

            # Hard-freeze the detector via requires_grad=False (not just
            # lr_mult=0 in the optimizer). Without this, autograd would
            # still compute grads for the detector's params on every
            # backward, and the head's gradient-checkpointed decoder layer
            # would do reentrant recompute that fires DDP's "marked ready
            # twice" check (with multiple frames-with-grad in the
            # sliding-window training loop).
            if self.freeze_detector:
                _motip_prefixes = ('id_dict.', 'pe_3d.', 'tracklet_former.',
                                   'id_decoder.')
                for name, param in self.named_parameters():
                    if not name.startswith(_motip_prefixes):
                        param.requires_grad = False


    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)


        return img_feats_reshaped


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            gt_instance_ids=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses

        # MOTIP: collect per-frame head outputs so compute_id_loss can run
        # over the full clip after this loop. Stays None when motip is off.
        motip_collected = [] if self.motip_cfg is not None else None

        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                data_t[key] = data[key][:, i]

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value

            # MOTIP: snapshot per-frame state needed for id loss
            if motip_collected is not None:
                motip_collected.append({
                    'outs': self._motip_last_outs,
                    'gt_bboxes_3d': gt_bboxes_3d[i],
                    'gt_labels_3d': gt_labels_3d[i],
                    'gt_instance_ids': gt_instance_ids[i] if gt_instance_ids is not None else None,
                    'ego_pose': data_t['ego_pose'],            # [B, 4, 4] lidar2global
                    'prev_exists': data_t.get('prev_exists', None),
                })

        if motip_collected is not None:
            self._motip_frame_data = motip_collected

        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        # MOTIP: cache the most-recent head outs so obtain_history_memory can
        # collect them per-frame without changing this method's return type.
        if self.motip_cfg is not None:
            self._motip_last_outs = outs

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d)

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            keys_to_transpose = ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
                                 'gt_labels', 'centers2d', 'depths', 'img_metas']
            # MOTIP: gt_instance_ids is also a per-frame variable-length list,
            # transpose only if present (lets non-MOTIP runs stay unaffected).
            if 'gt_instance_ids' in data:
                keys_to_transpose.append('gt_instance_ids')
            for key in keys_to_transpose:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      gt_instance_ids=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)

        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore,
                        gt_instance_ids=gt_instance_ids, **data)

        # MOTIP id loss over the full clip
        if self.motip_cfg is not None and getattr(self, '_motip_frame_data', None):
            id_losses = self.compute_id_loss(self._motip_frame_data)
            losses.update(id_losses)
            self._motip_frame_data = None  # release references

        return losses
  
  
    # ══════════════════════════════════════════════════════════════════
    # MOTIP integration
    # ══════════════════════════════════════════════════════════════════

    def _bbox_to_pe_input(self, bbox):
        """StreamPETR bbox [cx,cy,cz,w_log,l_log,h_log,sin,cos,vx,vy] →
        PE_3D input [x,y,z,w_log,l_log,h_log,θ,vx,vy] (9 dims).
        """
        theta = torch.atan2(bbox[:, 6:7], bbox[:, 7:8])
        return torch.cat([bbox[:, 0:6], theta, bbox[:, 8:10]], dim=-1)

    def _transform_bbox_to_current(self, bbox, src_l2g, dst_l2g):
        """Transform a [M, 10] StreamPETR-format bbox from src lidar coords
        into dst lidar coords. Position, yaw, and velocity are rotated; w/l/h
        (log space) are passed through. Out-of-place to keep gradients clean.
        """
        device = bbox.device
        src_l2g = src_l2g.to(device)
        dst_l2g = dst_l2g.to(device)
        T = torch.linalg.inv(dst_l2g) @ src_l2g  # src lidar -> dst lidar
        R = T[:3, :3]
        t = T[:3, 3]

        pos_dst = bbox[:, 0:3] @ R.T + t  # [M, 3]

        sin_y, cos_y = bbox[:, 6], bbox[:, 7]
        theta = torch.atan2(sin_y, cos_y)
        dyaw = torch.atan2(R[1, 0], R[0, 0])
        theta_dst = theta + dyaw

        vel_dst = bbox[:, 8:10] @ R[:2, :2].T  # [M, 2]

        return torch.cat([
            pos_dst,
            bbox[:, 3:6],
            torch.sin(theta_dst).unsqueeze(-1),
            torch.cos(theta_dst).unsqueeze(-1),
            vel_dst,
        ], dim=-1)

    def compute_id_loss(self, frame_data_list):
        """MOTIP cross-entropy ID loss over a clip.

        frame_data_list: list of T dicts (one per clip frame). Each carries:
            outs            : head output dict (query_feat / cls / bbox / dn_mask_dict)
            gt_bboxes_3d    : list[B] of LiDARInstance3DBoxes for this frame
            gt_labels_3d    : list[B] of Tensor labels
            gt_instance_ids : list[B] of int Tensor (per-detection int IDs)
            ego_pose        : [B, 4, 4] lidar2global at this frame
            prev_exists     : [B] (1=continuing scene, 0=scene start)

        Strategy A + decision 1-B: matching is run on all 428 queries (the
        full set, not just the fresh 300). Hungarian matching is independent
        of the detector's own loss matching, but uses the same assigner /
        sampler / cost weights so the GT->query assignment is consistent.
        """
        import torch.nn.functional as F

        T = len(frame_data_list)
        if T < 2:
            return {}  # need at least one context frame

        device = frame_data_list[0]['outs']['query_feat'].device
        B = frame_data_list[0]['outs']['query_feat'].shape[0]
        K = self.motip_cfg.get('num_ids', 50)
        det_frozen = self.freeze_detector

        head = self.pts_bbox_head
        match_costs = head.match_costs
        match_with_velo = head.match_with_velo

        # ── Step 1: Hungarian match each (b, t) ──
        matched = [[None] * T for _ in range(B)]
        for t in range(T):
            fd = frame_data_list[t]
            outs = fd['outs']
            query_feat = outs['query_feat']                  # [B, Q, C]
            cls_pred = outs['all_cls_scores'][-1]            # [B, Q, num_cls]
            bbox_pred = outs['all_bbox_preds'][-1]           # [B, Q, 10]
            for b in range(B):
                gt_bboxes = fd['gt_bboxes_3d'][b]
                gt_labels = fd['gt_labels_3d'][b]
                gt_ids = fd['gt_instance_ids'][b] if fd['gt_instance_ids'] is not None else None

                if gt_ids is None or len(gt_labels) == 0 or len(gt_ids) == 0:
                    continue

                gt_tensor = torch.cat(
                    [gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]], dim=1
                ).to(device)
                if not isinstance(gt_labels, torch.Tensor):
                    gt_labels = torch.tensor(gt_labels, dtype=torch.long, device=device)
                else:
                    gt_labels = gt_labels.to(device)
                if not isinstance(gt_ids, torch.Tensor):
                    gt_ids = torch.tensor(gt_ids, dtype=torch.long, device=device)
                else:
                    gt_ids = gt_ids.to(device)

                ar = head.assigner.assign(
                    bbox_pred[b], cls_pred[b],
                    gt_tensor, gt_labels,
                    None, match_costs, match_with_velo,
                )
                sr = head.sampler.sample(ar, bbox_pred[b], gt_tensor)
                if len(sr.pos_inds) == 0:
                    continue

                feat_b = query_feat[b, sr.pos_inds]
                bbox_b = bbox_pred[b, sr.pos_inds]
                if det_frozen:
                    feat_b = feat_b.detach()
                    bbox_b = bbox_b.detach()

                matched[b][t] = {
                    'feat': feat_b,
                    'bbox': bbox_b,
                    'raw_ids': gt_ids[sr.pos_assigned_gt_inds],
                    'lidar2global': fd['ego_pose'][b],
                }

        # ── Find scene boundaries per batch using prev_exists ──
        # scene_start[b] = first frame index of the current scene for batch b.
        # Frames before scene_start belong to a previous scene and must be
        # excluded from both ID re-indexing and context building.
        scene_start = [0] * B
        for b in range(B):
            for t in range(T):
                pe = frame_data_list[t].get('prev_exists')
                if pe is not None:
                    val = pe[b] if pe.dim() > 0 else pe
                    if val.item() == 0:
                        scene_start[b] = t

        # ── Step 2: per-batch ID re-indexing + per-frame loss ──
        loss_terms = []
        total_correct = 0
        total_predicted = 0
        total_matched = 0

        for b in range(B):
            b_matched = matched[b]
            ss = scene_start[b]

            # Only use frames from the current scene (>= scene_start)
            scene_frames = [b_matched[t] for t in range(ss, T)
                            if b_matched[t] is not None]
            if len(scene_frames) == 0:
                continue

            # Map raw int IDs to a per-clip 0..n_unique-1 space, then random-
            # permute slot assignments so the model can't memorise slot ↔
            # appearance mappings.
            all_raw = torch.cat([m['raw_ids'] for m in scene_frames])
            unique_ids, inv_all = all_raw.unique(return_inverse=True)
            n_unique = len(unique_ids)

            if n_unique <= K:
                perm = torch.randperm(K, device=device)[:n_unique]
                mapped = perm[inv_all]
            else:
                mapped = inv_all.clone()
                mapped[mapped >= K] = K

            # Distribute back to per-frame
            offset = 0
            for m in scene_frames:
                n = len(m['raw_ids'])
                m['mapped_ids'] = mapped[offset:offset + n]
                offset += n

            # ── For each target frame t (ss+1..T-1), build context from ss..t-1 ──
            for t in range(ss + 1, T):
                cur = b_matched[t]
                if cur is None:
                    continue

                cur_l2g = cur['lidar2global']

                ctx_tracklets = []
                ctx_id_lists = []

                for c in range(ss, t):
                    ctx = b_matched[c]
                    if ctx is None:
                        continue

                    # ctx → cur LiDAR coordinate transform (so PE stays
                    # consistent for static objects across frames)
                    ctx_bbox_cur = self._transform_bbox_to_current(
                        ctx['bbox'], ctx['lidar2global'], cur_l2g)
                    ctx_pe_in = self._bbox_to_pe_input(ctx_bbox_cur)
                    ctx_pe = self.pe_3d(ctx_pe_in[:, :7], ctx_pe_in[:, 7:9])

                    ctx_feat = ctx['feat']
                    ctx_ids = ctx['mapped_ids']

                    # Trajectory augmentation (out-of-place clones to keep
                    # the autograd graph clean even though feat is detached
                    # in frozen mode)
                    if self.training:
                        ctx_feat = ctx_feat.clone()
                        ctx_pe = ctx_pe.clone()
                        ctx_ids = ctx_ids.clone()
                        M = len(ctx_ids)
                        if M > 1 and torch.rand(1).item() < 0.5:
                            n_drop = max(1, M // 4)
                            drop_idx = torch.randperm(M, device=device)[:n_drop]
                            mask = torch.ones(M, 1, device=device)
                            mask[drop_idx] = 0.0
                            ctx_feat = ctx_feat * mask
                            ctx_pe = ctx_pe * mask
                        if M >= 2 and torch.rand(1).item() < 0.5:
                            ij = torch.randperm(M, device=device)[:2].tolist()
                            i, j = ij[0], ij[1]
                            tmp = ctx_ids[i].clone()
                            ctx_ids[i] = ctx_ids[j]
                            ctx_ids[j] = tmp

                    valid = ctx_ids < K
                    if valid.sum() == 0:
                        continue

                    valid_ids = ctx_ids[valid]
                    ctx_id_lists.append(valid_ids)
                    ctx_id_emb = self.id_dict.get_id_embedding(valid_ids)
                    tracklet = self.tracklet_former.form_tracklet(
                        ctx_feat[valid], ctx_pe[valid], ctx_id_emb)
                    ctx_tracklets.append(tracklet)

                if len(ctx_tracklets) == 0:
                    continue

                context = torch.cat(ctx_tracklets, dim=0).unsqueeze(0)  # [1, N_ctx, 3C]

                cur_pe_in = self._bbox_to_pe_input(cur['bbox'])
                cur_pe = self.pe_3d(cur_pe_in[:, :7], cur_pe_in[:, 7:9])
                spec = self.id_dict.get_special_token(len(cur['feat']))
                queries = self.tracklet_former.form_tracklet(
                    cur['feat'], cur_pe, spec).unsqueeze(0)

                # GT label for each cur query: its mapped ID if that ID
                # appeared in the context (any earlier frame), otherwise
                # newborn (K). Use torch.isin to avoid a Python loop +
                # GPU↔CPU sync per item (one of the SparseBEV-side perf bugs).
                ctx_id_tensor = torch.cat(ctx_id_lists)
                in_ctx = torch.isin(cur['mapped_ids'], ctx_id_tensor)
                gt_ids = torch.where(
                    in_ctx,
                    cur['mapped_ids'],
                    torch.full_like(cur['mapped_ids'], K),
                )

                id_logits = self.id_decoder(queries, context)  # [1, N_q, K+1]
                id_loss = F.cross_entropy(id_logits.squeeze(0), gt_ids)
                loss_terms.append(id_loss)
                total_matched += len(cur['feat'])

                with torch.no_grad():
                    preds = id_logits.squeeze(0).argmax(dim=-1)
                    total_correct += (preds == gt_ids).sum().item()
                    total_predicted += len(gt_ids)

        if len(loss_terms) == 0:
            # No MOTIP learning signal in this clip. Possible causes:
            #   - every frame had zero GT (very quiet scene)
            #   - Hungarian matching produced zero positives in every frame
            #   - matches existed but no (target, context) pair was valid
            # Emit a zero loss that touches *every* MOTIP submodule's params
            # so DDP sees the same parameter usage pattern as a normal iter
            # (otherwise find_unused_parameters=False crashes the next iter
            # with "Expected to have finished reduction in the prior
            # iteration"). Multiplying by 0 makes the gradient exactly zero,
            # so this path is purely a structural no-op.
            zero_loss = sum(
                p.sum()
                for module in (self.id_dict, self.pe_3d,
                               self.tracklet_former, self.id_decoder)
                for p in module.parameters()
            ) * 0.0
            return {
                'loss_id': zero_loss,
                'id_acc': torch.zeros((), device=device),
                'num_matched': torch.zeros((), device=device),
            }

        loss_id = torch.stack(loss_terms).mean() * self.id_loss_weight
        return {
            'loss_id': loss_id,
            'id_acc': torch.tensor(total_correct / max(total_predicted, 1), device=device),
            'num_matched': torch.tensor(total_matched / max(len(loss_terms), 1), device=device),
        }

    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # Attach MOTIP tracking features to results for extraction
        if self.motip_cfg is not None:
            bbox_results[0]['query_feat'] = outs['query_feat'][0].cpu()
            bbox_results[0]['bbox_raw'] = outs['all_bbox_preds'][-1][0].cpu()
            bbox_results[0]['cls_scores'] = outs['all_cls_scores'][-1][0].cpu()
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    