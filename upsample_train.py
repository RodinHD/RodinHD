"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.dataset.upsample_dataset_train import load_data
from pretrained_diffusion.resample import create_named_schedule_sampler
from pretrained_diffusion.upsample_train_util import TrainLoop
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,)
import deepspeed

import torch
import torch.utils.cpp_extension

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    torch.cuda.set_device(dist_util.dev())
     
    options = args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    print(args)
    model, diffusion = create_model_and_diffusion(**options)
   
    logger.configure(args.exp_name)
    options = args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    if  args.model_path:
        print('loading model from: ', args.model_path)
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")
        msg = model.load_state_dict(model_ckpt, strict=False)

    model.to(dist_util.dev())
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
########### dataset selection 
    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        low_res_root=args.low_res_root,
        train=True,
        low_res=args.super_res,
        uncond_p = args.uncond_p,
        mode = args.mode,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        strong_degradation=args.strong_degradation,
        no_degradation=args.no_degradation,
        num_patches=args.num_patches,
        txt_file=args.txt_file,
        pre_interpolate=True,
        image_root=args.image_root,
        latent_root=args.latent_root,
    )

    logger.log("training super-resolution model...")
    TrainLoop(
        args,
        model,
        options,
        diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        schedule_sampler=schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune_decoder=args.finetune_decoder,
        mode=args.mode,
        use_renderer=args.use_renderer,
        use_vgg=args.use_vgg,
        use_gan=args.use_gan,
        use_tv=args.use_tv,
        image_size = args.image_size,
        use_tensorboard=args.use_tensorboard,
        render_weight=args.render_weight,
        render_lpips_weight=args.render_lpips_weight,
        patch_size=args.patch_size,
        scale=args.scale,
        loss_type=args.loss_type,
        predict_type=args.predict_type,
    ).run_loop()

 
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
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        use_checkpoint=False,
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder=False,
        mode="",
        use_vgg=True,
        use_gan=False,
        use_tv=False,
        use_renderer="",
        use_tensorboard=True,
        n_feats=64,
        start_idx=-1,
        end_idx=-1,
        no_degradation=False,
        strong_degradation=False,
        render_weight=1.0,
        render_lpips_weight=0.5,
        patch_size=64,
        use_scale_shift_norm=False,
        txt_file="",
        scale=1.0,
        loss_type="l2",
        predict_type="xstart",
        decoder_type="default",
        num_patches=1,
        image_root="",
        latent_root="",
        )
 
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--ch_mult', dest='ch_mult', nargs='+', type=int)
    parser.add_argument('--low_res_root', dest='low_res_root', nargs='+', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser = deepspeed.add_config_arguments(parser)
 
    return parser


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.utils.cpp_extension.CUDA_HOME = "/usr/local/cuda/"
    print("cuda home: ", torch.utils.cpp_extension.CUDA_HOME)
    main()
