import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_seq, txt_seq):
        if txt_seq.shape[0] == 1 and img_seq.shape[0] > 1:
            txt_seq = txt_seq.expand(img_seq.shape[0], -1, -1)
        elif txt_seq.shape[0] != img_seq.shape[0]:
            raise ValueError(
                f"Batch mismatch: img={img_seq.shape[0]}, text={txt_seq.shape[0]}"
            )

        attn_out, attn_weights = self.attn(
            query=img_seq,
            key=txt_seq,
            value=txt_seq
        )

        fused = self.norm(img_seq + attn_out)
        return fused, attn_weights