import torch
import torch.nn as nn

class TrackletFormer(nn.Module):
    """τ = concat(f_obj, PE_3D, i_k) + TE(Δt)  →  [N, 3C].

    NOTE: SparseBEV's compute_id_loss never passes `rel_timestep`, so
    `temporal_embed` currently sits as an unused (but allocated) parameter.
    Left in place for now to avoid breaking checkpoint compatibility with
    SparseBEV-trained MOTIP weights. Drop it once we know we won't reuse
    those weights.
    """
    def __init__(self, embed_dim=256, max_temporal=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_embed = nn.Embedding(max_temporal, 3 * embed_dim)

    def form_tracklet(self, obj_embedding, pe_3d, id_embedding, rel_timestep=None):
        """
        obj_embedding: [N, C]  — SparseBEV query output
        pe_3d: [N, C]          — 3D positional encoding
        id_embedding: [N, C]   — ID dictionary embedding
        rel_timestep: [N]      — 현재 frame 대비 상대 offset
        Returns: [N, 3C=768]
        """
        tracklet = torch.cat([obj_embedding, pe_3d, id_embedding], dim=-1)
        if rel_timestep is not None:
            tracklet = tracklet + self.temporal_embed(rel_timestep)
        return tracklet