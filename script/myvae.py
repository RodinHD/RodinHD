from diffusers import AutoencoderKL
from diffusers.models import vae
from typing import Dict, Optional, Tuple, Union

class VAEKL(AutoencoderKL):
    def __init__(self,
                in_channels: int = 3,
                out_channels: int = 3,
                down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                block_out_channels: Tuple[int] = (64,),
                layers_per_block: int = 1,
                act_fn: str = "silu",
                latent_channels: int = 4,
                norm_num_groups: int = 32,
                sample_size: int = 32,
                scaling_factor: float = 0.18215,
                force_upcast: float = True):
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels, layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, scaling_factor, force_upcast)
        self.encoder = MyEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
    
    def encode_feat(self, x):
        sample, feat_list = self.encoder(x)
        return sample, feat_list[:3]
    

class MyEncoder(vae.Encoder):
    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        feat_list = []
        for down_block in self.down_blocks:
            sample = down_block(sample)
            feat_list.append(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample, feat_list