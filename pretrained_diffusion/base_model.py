import torch as th
import torch.nn as nn
from .nn import timestep_embedding, AttentionPooling, normalization
from .unet import UNetModel

class BaseDiffusion(nn.Module):
    def __init__(
        self,
        xf_width,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout,
        channel_mult,
        use_fp16,
        num_heads,
        num_heads_upsample,
        num_head_channels,
        use_scale_shift_norm,
        resblock_updown, 
        in_channels=32,  
    ):
        super().__init__()
        
        self.in_channels = in_channels
        class_name = BaseMultiscaleVAELatentUNet
        self.decoder = class_name(
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            encoder_channels=xf_width
        )

    def forward(self, xt, timesteps, ref=None, vae_ms_feature=None):
        pred = self.decoder(xt, timesteps, ref, vae_ms_feature)
        return pred

class BaseMultiscaleVAELatentUNet(UNetModel):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs,  
    ):
        super().__init__(in_channels, *args, **kwargs)
        self.dtype = th.float32
        patch_size = 16
        encoder_dim = 64
        att_pool_heads = 8
        self.scaling_factor = 0.13025
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=encoder_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        
        self.encoder_pooling = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            AttentionPooling(att_pool_heads, encoder_dim),
            nn.Linear(encoder_dim, self.model_channels * 4),
            nn.LayerNorm(self.model_channels * 4)
        )

        self.norm32 = normalization(512)
        self.norm64 = normalization(256)
        self.norm128 = normalization(128)
        self.vae_ms_feature_proj_32 = nn.Conv2d(512, self.encoder_channels, kernel_size=4, stride=4, bias=False)
        self.vae_ms_feature_proj_64 = nn.Conv2d(256, self.encoder_channels, kernel_size=8, stride=8, bias=False)
        self.vae_ms_feature_proj_128 = nn.Conv2d(128, self.encoder_channels, kernel_size=16, stride=16, bias=False)
  
    def forward(self, x, timesteps, latent_outputs, vae_ms_feature):
        '''
        latent_outputs: dict {'last_hidden_state': tensor, 'pooler_output': tensor}
        '''
        input_type = x.dtype
        x = x.type(self.dtype)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(self.dtype))
        
        latent_outputs = (latent_outputs * self.scaling_factor).type(self.dtype)
        latent_outputs_emb = self.conv1(latent_outputs)  # shape = [*, width, grid, grid]
        latent_outputs_emb = latent_outputs_emb.reshape(latent_outputs_emb.shape[0], latent_outputs_emb.shape[1], -1)  # shape = [*, width, grid ** 2]
        latent_outputs_emb = latent_outputs_emb.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        encoder_pool = self.encoder_pooling(latent_outputs_emb)
        emb = emb + encoder_pool.to(emb)

        vae_feature_32 = self.vae_ms_feature_proj_32(self.norm32(vae_ms_feature[2]).type(self.dtype))
        vae_feature_32 = vae_feature_32.reshape(vae_feature_32.shape[0], vae_feature_32.shape[1], -1)  # shape: (N, C, L)
        vae_feature_64 = self.vae_ms_feature_proj_64(self.norm64(vae_ms_feature[1]).type(self.dtype))
        vae_feature_64 = vae_feature_64.reshape(vae_feature_64.shape[0], vae_feature_64.shape[1], -1) # shape: (N, C, L)
        vae_feature_128 = self.vae_ms_feature_proj_128(self.norm128(vae_ms_feature[0]).type(self.dtype))
        vae_feature_128 = vae_feature_128.reshape(vae_feature_128.shape[0], vae_feature_128.shape[1], -1) # shape: (N, C, L)
        encoder_out_feature = {32: vae_feature_128, 16: vae_feature_64, 8: vae_feature_32}

        h = x.type(self.dtype)
        for module in self.input_blocks:
            encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
            h = module(h, emb, encoder_out)  
            hs.append(h)
        
        encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
        h = self.middle_block(h, emb, encoder_out)

        for module in self.output_blocks:
            encoder_out = encoder_out_feature[h.shape[-2]] if h.shape[-2] in encoder_out_feature else None
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, encoder_out)
        h = h.type(input_type)
        h = self.out(h)
        return h.type(input_type)