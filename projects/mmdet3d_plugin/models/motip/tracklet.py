import torch
import torch.nn as nn


class TrackletFormer(nn.Module):
    """τ = concat(f_obj, PE_3D, i_k)  →  [N, 3C].

    Originally there was a `temporal_embed` nn.Embedding here, conditional
    on a `rel_timestep` argument that nothing actually passes. With DDP +
    `find_unused_parameters=False` (required by gradient checkpointing in
    StreamPETR's head), an unused parameter raises an error, so we drop it
    entirely. If a temporal embedding is needed in the future, re-add it
    behind a config flag and route `rel_timestep` through compute_id_loss.
    """
    def __init__(self, embed_dim=256, max_temporal=20):
        super().__init__()
        self.embed_dim = embed_dim
        # max_temporal kept in the signature for backwards-compatible config

    def form_tracklet(self, obj_embedding, pe_3d, id_embedding, rel_timestep=None):
        """
        obj_embedding: [N, C]
        pe_3d: [N, C]
        id_embedding: [N, C]
        rel_timestep: kept for API compatibility but currently ignored.
        Returns: [N, 3C]
        """
        return torch.cat([obj_embedding, pe_3d, id_embedding], dim=-1)
