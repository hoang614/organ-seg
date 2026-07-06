import torch.nn as nn

from .image_encoder import ImageEncoder3D
from .text_encoder import TextEncoder
from .fusion import CrossModalFusion
from .decoder import Decoder3D

class MultiModalSegModel(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        feature_size=48,
        embed_dim=768,
        text_model_name="dmis-lab/biobert-base-cased-v1.1",
        num_heads=8,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder3D(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            output_dim=embed_dim
        )
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            output_dim=embed_dim
        )
        self.fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.decoder = Decoder3D(
            in_channels=embed_dim,
            out_channels=out_channels,
            feature_size=feature_size
        )

    def forward(self, image, text_inputs):
        img_seq, spatial_shape, skips = self.image_encoder(image)   # img_seq: [B, N, C]
        txt_seq = self.text_encoder(text_inputs)             # txt_seq: [B, T, C]

        fused_seq, _ = self.fusion(img_seq, txt_seq)

        B, N, C = fused_seq.shape
        Dp, Hp, Wp = spatial_shape

        fused_vol = fused_seq.transpose(1, 2).contiguous().view(B, C, Dp, Hp, Wp)
        output = self.decoder(fused_vol, skips)

        return output