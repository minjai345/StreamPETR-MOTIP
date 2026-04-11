_base_ = ['./stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.py']

# Override: replace flash attention with vanilla PETRMultiheadAttention.
# flash_attn is not installed in this env (torch 2.0.0+cu118).
# Per maintainer (issue #23), comment out FlashMHA import + use non-flash variant.
# Numerically equivalent — checkpoint loads correctly because parameter names match.

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                    ],
                ),
            ),
        ),
    ),
)
