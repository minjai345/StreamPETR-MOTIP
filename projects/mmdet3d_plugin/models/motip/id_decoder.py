import torch
import torch.nn as nn

class IDDecoder(nn.Module):
    """Transformer Decoder for K+1 class ID prediction.
    Query: 현재 frame detections (i_spec 포함)
    Key/Value: historical tracklets (할당된 ID 포함)
    """
    def __init__(self, d_model=768, nhead=8, num_layers=6,
                 dim_feedforward=2048, num_ids=50, dropout=0.1):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.id_head = nn.Linear(d_model, num_ids + 1)

    def forward(self, queries, context, query_mask=None, context_mask=None):
        """
        queries: [B, N_q, 3C]   — 현재 detections + i_spec
        context: [B, N_kv, 3C]  — historical tracklets
        Returns: [B, N_q, K+1]  — ID logits
        """
        # PyTorch 1.13: attention mask는 bool 타입 사용
        decoded = self.decoder(
            tgt=queries,
            memory=context,
            tgt_key_padding_mask=query_mask,
            memory_key_padding_mask=context_mask
        )
        return self.id_head(decoded)