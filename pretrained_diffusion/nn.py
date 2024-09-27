"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils import checkpoint


def gelu(x):
    return 0.5 * x * (1.0 + th.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


@th.jit.script
def gelu_jit(x):
    """OpenAI's gelu implementation."""
    return gelu(x)


class GELUJit(th.nn.Module):
    def forward(self, input):
        return gelu_jit(input)


def get_activation(activation):
    if activation == 'silu':
        return nn.SiLU()
    elif activation == 'gelu_jit':
        return GELUJit()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'unknown activation type {activation}')

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        weight_dtype = self.weight.dtype  
        if weight_dtype != x.dtype:  
            x = x.to(weight_dtype)
        y = super().forward(x).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, swish=0.0):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish)


def timestep_embedding(timesteps, dim, max_period=10000, dtype=None):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if dtype is None:
        dtype = th.float32
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device, dtype=dtype)
    args = timesteps[:, None].type(dtype) * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class Conv3DAware2(nn.Module):
    def __init__(
        self, C, H, W):
        super(Conv3DAware2, self).__init__()
        self.up_down    = nn.Conv2d(C,1*C,(H,1), stride=1)
        self.left_right = nn.Conv2d(C,1*C,(1,W), stride=1)

        self.out_layers1 = nn.Conv2d(3*C,C,3, stride=1, padding=1)
        self.out_layers2 = nn.Conv2d(3*C,C,3, stride=1, padding=1)
        self.out_layers3 = nn.Conv2d(3*C,C,3, stride=1, padding=1)

    def forward(self, triplane):
        B, C, H, W = triplane.shape
     
        W = W//3
        assert B == 1 
        triplane = th.stack([triplane[:,:,:,:W], triplane[:,:,:,W:2*W], triplane[:,:,:,2*W:]], 0) 
        triplane = triplane.view(3,C,H,W)

        ### group
        xoy = triplane[0].unsqueeze(0) 
        xoz = triplane[1].unsqueeze(0) 
        yoz = triplane[2].unsqueeze(0) 

        xoy2ox = self.up_down(xoy).expand(-1,-1,H,-1) 
        xoz2ox = self.up_down(xoz).expand(-1,-1,H,-1) 
        yoz2oy = self.up_down(yoz).expand(-1,-1,H,-1) 

        xoy2oy = self.left_right(xoy).expand(-1,-1,-1,W) 
        xoz2oz = self.left_right(xoz).expand(-1,-1,-1,W) 
        yoz2oz = self.left_right(yoz).expand(-1,-1,-1,W) 

        xoy_ = th.cat([xoy, xoz2ox, yoz2oy.permute(0,1,3,2)], 1)   
        xoz_ = th.cat([xoy2ox, xoz, yoz2oz], 1)
        yoz_ = th.cat([xoy2oy.permute(0,1,3,2), xoz2oz, yoz], 1)

        ### conv
        xoy_ = self.out_layers1(xoy_)
        xoz_ = self.out_layers2(xoz_)
        yoz_ = self.out_layers3(yoz_)
        h    = th.cat([xoy_, xoz_, yoz_], -1)
  
        return h

class Conv3DAware(nn.Module):
    def __init__(
        self, C, C_out, dtype=th.float32):
        super(Conv3DAware, self).__init__()
     
        self.up_down    = nn.AdaptiveAvgPool2d((1, None))
        self.left_right = nn.AdaptiveAvgPool2d((None, 1))

        self.out_layers1 = zero_module(nn.Conv2d(3*C,C_out,3, stride=1, padding=1, dtype=dtype))
        self.out_layers2 = zero_module(nn.Conv2d(3*C,C_out,3, stride=1, padding=1, dtype=dtype))
        self.out_layers3 = zero_module(nn.Conv2d(3*C,C_out,3, stride=1, padding=1, dtype=dtype))

    def forward(self, triplane):
        B, C, H, W = triplane.shape
        W = W//3
    
        triplane = th.stack([triplane[:,:,:,:W], triplane[:,:,:,W:2*W], triplane[:,:,:,2*W:]], 1) 
        triplane = triplane.view(B,3,C,H,W)

        ### group
        xoy = triplane[:,0] 
        xoz = triplane[:,1] 
        yoz = triplane[:,2] 

        xoy2ox = self.up_down(xoy).expand(-1,-1,H,-1) 
        xoz2ox = self.up_down(xoz).expand(-1,-1,H,-1) 
        yoz2oy = self.up_down(yoz).expand(-1,-1,H,-1) 
        
        xoy2oy = self.left_right(xoy).expand(-1,-1,-1,W) 
        xoz2oz = self.left_right(xoz).expand(-1,-1,-1,W) 
        yoz2oz = self.left_right(yoz).expand(-1,-1,-1,W) 

        xoy_ = th.cat([xoy, xoz2ox, yoz2oy.permute(0,1,3,2)], 1)   
        xoz_ = th.cat([xoy2ox, xoz, yoz2oz], 1)
        yoz_ = th.cat([xoy2oy.permute(0,1,3,2), xoz2oz, yoz], 1)

        ### conv
        xoy_ = self.out_layers1(xoy_)
        xoz_ = self.out_layers2(xoz_)
        yoz_ = self.out_layers3(yoz_)
        h    = th.cat([xoy_, xoz_, yoz_], -1)

        return h
 

class AttentionPooling(nn.Module):

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(th.randn(1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=self.dtype)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs*self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(x.dtype)
        x = th.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = th.einsum(
            'bct,bcs->bts', q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = th.einsum('bts,bcs->bct', weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
        out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** th.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = th.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return th.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3, periodic_fns=None):
    if periodic_fns is None:
        periodic_fns = [th.sin, th.cos]
    embed_kwargs = {
        'include_input': False,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': periodic_fns,
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


def make_grid(H, W, device):
    # Create 1D tensors for each dimension  
    h = th.arange(H)  
    w = th.arange(W)  
    # d = torch.arange(D, dtype=torch.float32)  
    
    # Generate the meshgrid  
    h_grid, w_grid = th.meshgrid(h, w)  
    
    # Stack the tensors along a new dimension  
    coordinates = th.stack((h_grid, w_grid), dim=-1) / H * 2 - 1

    sample_coordinates = coordinates.view(-1, 2).unsqueeze(0)
    return sample_coordinates.to(device)
