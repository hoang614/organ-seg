from monai.networks.nets import SwinUNETR
import torch.nn as nn

class ImageEncoder3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_size=48, output_dim=768):
        super().__init__()
        self.swin_vit = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            window_size=7,
            dropout_path_rate=0.1,
            use_checkpoint=True,
        ).swinViT

        self.proj = nn.Linear(feature_size * 8, output_dim)

    def forward(self, x):
        # x: [B, 1, D, H, W]
        feats = self.swin_vit(x)

        skip0 = feats[0]
        skip1 = feats[1]
        skip2 = feats[2]
        skip3 = feats[3]

        bottleneck_feat = feats[3]  # [B, C, D', H', W']

        B, C, D_sup, H_sup, W_sup = bottleneck_feat.shape
        x_seq = bottleneck_feat.view(B, C, -1).transpose(1, 2) # [B, D' * H' * W', C]
        x_seq = self.proj(x_seq)

        skips = [skip0, skip1, skip2, skip3]

        return x_seq, (D_sup, H_sup, W_sup), skips