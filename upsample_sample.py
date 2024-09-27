"""
Train a diffusion model on images.
"""
import argparse
import numpy as np
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.dataset.upsample_dataset_test import load_data
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_gaussian_diffusion,
)
from pretrained_diffusion.ddim import DDIMSampler
import torch
import os
import torch as th
import torch.distributed as dist

def main():
    args = create_argparser().parse_args()

    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = True

    dist_util.setup_dist()

    options = args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    model, _ = create_model_and_diffusion(**options)
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    logger.configure(args.exp_name)
    if dist.get_rank() == 0:
        logger.save_args(options)
 
    if  args.model_path:
        print('loading model from: ', args.model_path)
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")
        model.load_state_dict(model_ckpt, strict=True )

    model.to(dist_util.dev())
    model.eval()
 
    glide_options = options
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        learn_sigma=glide_options["pretrained_learn_sigma"] if glide_options["pretrained_learn_sigma"] else glide_options["learn_sigma"],
        noise_schedule=glide_options["noise_schedule"],
        predict_xstart=glide_options["predict_xstart"],
        rescale_timesteps=glide_options["rescale_timesteps"],
        rescale_learned_sigmas=glide_options["rescale_learned_sigmas"],
        timestep_respacing=str(glide_options["diffusion_steps"]),
        predict_type=args.predict_type
    )
    sampler = DDIMSampler(model, eval_diffusion, schedule="linear", parameterization=args.predict_type, device=dist_util.dev())

    logger.log("creating data loader...")
    val_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=False,
        deterministic=True,
        low_res=args.super_res,
        uncond_p=0. ,
        mode=args.mode,
        txt_file=args.txt_file,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        scale=args.scale,
        latent_root=args.latent_root,
    )

    logger.log("sampling...")
    lr_path = os.path.join(logger.get_dir(), 'LR')
    os.makedirs(lr_path,exist_ok=True)
    hr_path = os.path.join(logger.get_dir(), 'HR')
    os.makedirs(hr_path,exist_ok=True)    

    img_id = 0
    while (True):
        if img_id >= args.num_samples:
            break

        batch, model_kwargs = next(val_data)    
        batch = batch.to(dist_util.dev()) 
        
        uncond = torch.ones_like(model_kwargs["ref"])
        with th.no_grad():
            print("sampling triplane ", img_id)
            model_kwargs["ref"] = model_kwargs["ref"].to(dist_util.dev())
            unconditional_conditioning = {'ref': uncond.to(dist_util.dev()), 'low_res': model_kwargs['low_res'].to(dist_util.dev()) * args.scale}
            model_kwargs['low_res'] = model_kwargs['low_res'].to(dist_util.dev()) * args.scale

            shape = [32, args.image_size, args.image_size*3]
            samples, _ = sampler.sample(S=int(args.sample_respacing),
                                        conditioning=None if args.sample_c <= 1. else {'ref': model_kwargs["ref"], 'low_res': model_kwargs['low_res']},
                                        batch_size=args.batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=args.sample_c,
                                        unconditional_conditioning=None if args.sample_c <= 1. else unconditional_conditioning,
                                        eta=args.eta,
                                        x_T=None,
                                        clip_denosied=True,
                                        model_kwargs={'ref': model_kwargs["ref"], 'low_res': model_kwargs['low_res']})

            name = model_kwargs['path'][0].split('/')[-1]
            samples = samples.cpu()

            for i in range(samples.size(0)):
                name = model_kwargs['path'][i].split('/')[-1].split('.')[0] + ".npy"
                out_path = os.path.join(hr_path, name)
                with open(out_path, 'wb') as f:
                    np.save(f, samples[i].detach().cpu().to(th.float16).numpy())     
                img_id += 1     


def create_argparser():
    defaults = dict(
        exp_name ="",
        data_dir="",
        val_data_dir="",
        model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=200,
        save_interval=20000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="10",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode= "",
        use_tv=False,
        start_idx=0,
        end_idx=10,
        scale=1.0,
        use_scale_shift_norm=False,
        predict_type="xstart",
        txt_file="",
        eta=0,
        latent_root="",
        seed=0,
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--ch_mult', dest='ch_mult', nargs='+', type=int)
    add_dict_to_argparser(parser, defaults)

    return parser 


if __name__ == "__main__":
    # set random seed
    main()
 
 