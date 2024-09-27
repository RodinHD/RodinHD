import os
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import math 
from .nn import linear, timestep_embedding, conv_nd, Conv3DAware, get_embedder, make_grid, AttentionPooling, zero_module
from .unet import TimestepEmbedSequential, TimestepBlock, CrossAttentionBlock
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .utils import checkpoint

 
class Upsampler(nn.Sequential):
    def __init__(self,  scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:     
            for _ in range(int(math.log(scale, 2))):
                m.append(conv_nd(2, n_feats, 4*n_feats, 3, padding=1))
                
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def nonlinearity(x):
    # swish
    return x*th.sigmoid(x)


def swish_0(x):
    return x * F.sigmoid(x * float(0))


def Normalize(in_channels, num_groups=32):
    return th.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv,):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = th.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        input_type = x.dtype
        if self.with_conv:
            pad = (0,1,0,1)
            x = th.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = th.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        if self.with_conv:
            self.conv = th.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        input_type = x.dtype
        x = self.up(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(TimestepBlock):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, use_3d_conv=False, use_checkpoint=False, use_scale_shift_norm=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = Normalize(in_channels)
        self.conv1 = th.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = th.nn.Linear(temb_channels,
                                             2 * out_channels if use_scale_shift_norm else out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = th.nn.Dropout(dropout)
        #TODO: add conv axis
        if use_3d_conv:
            self.conv2 = Conv3DAware(out_channels, out_channels)
        else:
            self.conv2 = th.nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = th.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = th.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)
    def forward(self, x, temb=None):
        return checkpoint(self._forward, (x, temb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            if not self.use_scale_shift_norm:
                h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]
            else:
                emb_out = self.temb_proj(nonlinearity(temb))[:,:,None,None]

        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = self.norm2(h)
            h = swish_0(h)
            h = h * (scale + 1) + shift
            h = self.dropout(h)
            h = self.conv2(h)
        else:
            h = self.norm2(h)
            h = nonlinearity(h)
            h = self.dropout(h)
            h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class TriplaneUpsampler(nn.Module):
    def __init__(self, scale=4, image_size=256, n_feats=64, n_resblocks=6, kernel_size=3, use_fp16=False, use_checkpoint=False, ch_mult=[1, 2], use_scale_shift_norm=False, use_3d_conv=True, condition_channels=32, dtype="32"):
        super(TriplaneUpsampler, self).__init__()

        self.n_feats = n_feats
        self.temb_ch = n_feats * 4
        self.num_res_blocks = 2
        self.num_resolutions = len(ch_mult)
        self.dtype = th.float16 if dtype=="16" else th.float32
        self.use_checkpoint = use_checkpoint
        self.attention_level = (2,)
        print("use_checkpoint", use_checkpoint)
        ch = n_feats

        in_ch_mult = [1, ] + list(ch_mult)
        resamp_with_conv = True
        use_3d_conv = use_3d_conv

        self.time_embed = nn.Sequential(
            linear(n_feats, n_feats*4),
            nn.SiLU(),
            linear(n_feats*4, n_feats*4),
        )

        self.input_layer = TimestepEmbedSequential(conv_nd(2, 32, n_feats, kernel_size, padding=(kernel_size//2)),
                           ResnetBlock(in_channels=n_feats, out_channels=ch*in_ch_mult[0], temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
        )

        # Build encoder
        self.down = nn.ModuleList()
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level == 0 and i_block == 0:
                    block_in += condition_channels
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0,
                                         use_3d_conv=use_3d_conv,
                                         use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out
            
            down = nn.Module()
            down.block = TimestepEmbedSequential(*block)
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
            self.down.append(down)

        # define body module
        m_body = []
        for _ in range(n_resblocks//2):
            m_body.extend([
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
                CrossAttentionBlock(block_in, num_head_channels=32, disable_self_attention=True, encoder_channels=self.temb_ch, dtype=self.dtype),
                ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            ])
        
        # Build decoder
        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                if i_level == 0 and i_block == self.num_res_blocks:
                    skip_in += condition_channels
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0,
                                         use_3d_conv=use_3d_conv,
                                         use_checkpoint=use_checkpoint, 
                                         use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out

            up = nn.Module()
            up.block = TimestepEmbedSequential(*block)
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
            self.up.insert(0, up) # prepend to get consistent order

        # define tail module
        m_tail = [
            ResnetBlock(in_channels=block_in+n_feats+condition_channels, out_channels=n_feats, temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            ResnetBlock(in_channels=n_feats, out_channels=32, temb_channels=self.temb_ch, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        ]
        
        patch_size = 16
        encoder_dim = 64
        att_pool_heads = 8
        self.scaling_factor = 0.13025
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=encoder_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        if encoder_dim != self.temb_ch:
            self.encoder_proj = nn.Linear(encoder_dim, self.temb_ch)
        else:
            self.encoder_proj = nn.Identity() 
        
        self.encoder_pooling = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            AttentionPooling(att_pool_heads, encoder_dim),
            nn.Linear(encoder_dim, n_feats * 4),
            nn.LayerNorm(n_feats * 4)
        )
        self.transformer_proj = nn.Identity()
        self.body = TimestepEmbedSequential(*m_body)
        self.tail = TimestepEmbedSequential(*m_tail)
        self.out = ResnetBlock(in_channels=32+condition_channels, out_channels=32, temb_channels=0, dropout=0, use_3d_conv=use_3d_conv, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_layer.apply(convert_module_to_f16)
        self.down.apply(convert_module_to_f16)
        self.body.apply(convert_module_to_f16)
        self.up.apply(convert_module_to_f16)
        self.tail.apply(convert_module_to_f16)
        if not self.rezero:
            self.out.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_layer.apply(convert_module_to_f32)
        self.down.apply(convert_module_to_f32)
        self.body.apply(convert_module_to_f32)
        self.up.apply(convert_module_to_f32)
        self.tail.apply(convert_module_to_f32)
        if not self.rezero:
            self.out.apply(convert_module_to_f32)

    def forward(self, xt, time_steps, ref, low_res=None):
        # lightweight encoder
        input_type = low_res.dtype
        b, c, h, w3 = low_res.shape
        low_res = F.interpolate(th.concat([low_res[..., :h], low_res[..., h:2*h], low_res[..., 2*h:]], 0), (xt.shape[-2], xt.shape[-2]), mode="bicubic")
        low_res = th.cat([low_res[0], low_res[1], low_res[2]], -1).unsqueeze(0)
        low_res = low_res.type(self.dtype) #.type(th.float32)
        xt = xt.type(self.dtype)
        emb = self.time_embed(timestep_embedding(time_steps, self.n_feats).to(self.dtype))

        latent_outputs = (ref * self.scaling_factor).type(self.dtype)
        latent_outputs_emb = self.conv1(latent_outputs)  # shape = [*, width, grid, grid]
        latent_outputs_emb = latent_outputs_emb.reshape(latent_outputs_emb.shape[0], latent_outputs_emb.shape[1], -1)  # shape = [*, width, grid ** 2]
        latent_outputs_emb = latent_outputs_emb.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        encoder_pool = self.encoder_pooling(latent_outputs_emb)
        emb = emb + encoder_pool.to(emb)

        encoder_out = self.encoder_proj(latent_outputs_emb)
        encoder_out = encoder_out.permute(0, 2, 1)  # NLC -> NCL
        encoder_out = encoder_out.type(self.dtype)
        
        x = self.input_layer(xt, emb)
        x = th.cat([x, low_res], dim=1)

        hs = [x]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], emb)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.body(h, emb, encoder_out)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](th.cat([h, hs.pop()], dim=1), emb)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # add low_res
        x = self.tail(th.cat([h, x], dim=1), emb)
        output = self.out(th.cat([x, low_res], dim=1))
 
        return output.type(input_type)
    