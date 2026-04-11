import torch
import torch.nn as nn

class IDDictionary(nn.Module):
    """K개의 learnable ID embedding + 1개의 newborn special token."""
    def __init__(self, num_ids=50, embed_dim=256):
        super().__init__()
        self.num_ids = num_ids
        self.embed_dim = embed_dim
        self.special_idx = num_ids
        self.embeddings = nn.Embedding(num_ids + 1, embed_dim)

    def get_id_embedding(self, id_indices):
        """id_indices: [N] → [N, C]"""
        return self.embeddings(id_indices)

    def get_special_token(self, batch_size):
        """newborn special token: [batch_size, C]"""
        idx = torch.full((batch_size,), self.special_idx,
                         dtype=torch.long, device=self.embeddings.weight.device)
        return self.embeddings(idx)