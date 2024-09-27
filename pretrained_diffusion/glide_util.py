import os
from typing import Tuple
from . import dist_util
import PIL
import numpy as np
import torch as th
from .script_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# Sample from the base model.

#@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_temp=0.997,
    mode = '',
    noise = None,
    predict_type='noise'
):

    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        learn_sigma=glide_options["pretrained_learn_sigma"] if glide_options["pretrained_learn_sigma"] else glide_options["learn_sigma"],
        noise_schedule=glide_options["noise_schedule"],
        predict_xstart=glide_options["predict_xstart"],
        rescale_timesteps=glide_options["rescale_timesteps"],
        rescale_learned_sigmas=glide_options["rescale_learned_sigmas"],
        timestep_respacing=prediction_respacing,
        predict_type=predict_type
    )
 

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
  
    model_kwargs = {}

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :32], model_out[:, 32:]
        
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
 
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)


    if upsample_enabled:
        model_kwargs['low_res'] = prompt['low_res'].to(dist_util.dev())
        model_kwargs['ref'] = prompt['ref'].to(dist_util.dev())
        if 'vae_ms_feature' in prompt:
            model_kwargs['vae_ms_feature'] = [feat.to(dist_util.dev()) for feat in prompt['vae_ms_feature']]

        noise = th.randn((batch_size, 32, side_y, side_x), device=device) * upsample_temp
        model_fn = glide_model # just use the base model, no need for CFG.
 
        samples = eval_diffusion.p_sample_loop(
        model_fn,
        (batch_size, 32, side_y, side_x),  # only thing that's changed
        noise=noise,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    else:
        if th.is_tensor(prompt['ref']):
            model_kwargs['ref'] = prompt['ref'].to(dist_util.dev())
        model_kwargs['ref'] = prompt['ref']
        
        model_fn = cfg_model_fn # so we use CFG for the base model.
        if noise is None:
            noise = th.randn((batch_size, 32, side_y, side_x), device=device) 
   
        noise = th.cat([noise, noise], 0)
        samples = eval_diffusion.p_sample_loop(
            model_fn,
            (batch_size*2, 32, side_y, side_x),  # only thing that's changed
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]    

    return samples
