"""
Train a diffusion model on images.
"""
import argparse
import numpy as np
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.dataset.base_dataset_test import load_data
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_gaussian_diffusion,
)
from pretrained_diffusion.base_train_util import TrainLoop
from pretrained_diffusion.ddim import DDIMSampler
import torch
import os
import torch as th
import torch.distributed as dist
from mpi4py import MPI


def main():
    args, _ = create_argparser()
    args = args.parse_args()
    dist_util.setup_dist()

    options = args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    model, diffusion = create_model_and_diffusion(**options)
    
    logger.configure(args.exp_name)
    if  args.model_path:
        print('loading model from ', args.model_path)
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
        latent_root=args.latent_root,
        ms_feature_root=args.ms_feature_root,
    )
 
    logger.log("sampling...")
    gt_path = os.path.join(logger.get_dir(), 'GT')
    os.makedirs(gt_path,exist_ok=True)
    lr_path = os.path.join(logger.get_dir(), 'LR')
    os.makedirs(lr_path,exist_ok=True)    

    l2_loss = th.nn.MSELoss()
    img_id = 0
    triplane_losses = []
    while (True):
        if img_id >= args.num_samples:
            break

        batch, model_kwargs = next(val_data)    
        uncond = torch.ones_like(model_kwargs["ref"])
        with th.no_grad():
            model_kwargs["ref"] = model_kwargs["ref"].to(dist_util.dev())
            model_kwargs['vae_ms_feature'] = [tensor.to(dist_util.dev()) for tensor in model_kwargs['vae_ms_feature']]
            unconditional_conditioning = {'ref': uncond.to(dist_util.dev()), 'vae_ms_feature': [th.ones_like(tensor).to(dist_util.dev()) for tensor in model_kwargs['vae_ms_feature']]}
            conditional_conditioning = {'ref': model_kwargs["ref"], 'vae_ms_feature': model_kwargs['vae_ms_feature']}

        with th.no_grad():   
            print("sampling triplane ", img_id)
            name = model_kwargs['path'][0].split('/')[-1].replace(".png", ".npy")
            shape = [32, args.image_size, args.image_size*3]
            samples_lr, _ = sampler.sample(S=int(args.sample_respacing),
                                        conditioning=conditional_conditioning,
                                        batch_size=args.batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=args.sample_c,
                                        unconditional_conditioning=unconditional_conditioning,
                                        eta=args.eta,
                                        clip_denosied=True,
                                        model_kwargs=None)

            samples_lr = samples_lr.cpu()

            for i in range(samples_lr.size(0)):
                name = model_kwargs['path'][i].split('/')[-1].split('.')[0] + ".npy" 
                out_path = os.path.join(lr_path, name)
                with open(out_path, 'wb') as f:
                    np.save(f, torch.concat([samples_lr[i][:, :, :args.image_size], samples_lr[i][:,:,args.image_size:args.image_size*2], samples_lr[i][:,:,args.image_size*2:]],0).reshape(3, 1, 32, args.image_size, args.image_size).detach().cpu().to(th.float16).numpy())    

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
        sample_respacing="",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode="",
        use_tv=False,
        start_idx=0,
        end_idx=10,
        predict_type='noise',
        eta=0,
        txt_file='',
        latent_root='',
        ms_feature_root='',
        )

    defaults_up = defaults
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--ch_mult', dest='ch_mult', nargs='+', type=int)
    add_dict_to_argparser(parser, defaults)

    defaults_up.update(model_and_diffusion_defaults(True))
    parser_up = argparse.ArgumentParser()
    add_dict_to_argparser(parser_up, defaults_up)

    return parser, parser_up


if __name__ == "__main__":
    main()
 
 