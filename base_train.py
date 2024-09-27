"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
from pretrained_diffusion import dist_util, logger
from pretrained_diffusion.dataset.base_dataset_train import load_data
from pretrained_diffusion.resample import create_named_schedule_sampler
from pretrained_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,)
from pretrained_diffusion.base_train_util import TrainLoop
import torch
import clip
from transformers import CLIPVisionModel

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    options = args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys())
    options['ch_mult'] = args.ch_mult
    model, diffusion = create_model_and_diffusion(**options)

    logger.configure(args.exp_name)
    options = args_to_dict(args)
    if dist.get_rank() == 0:
        logger.save_args(options)

    if  args.model_path:
        print('loading model')
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")
        msg = model.load_state_dict(model_ckpt, strict=False )
    
    model.to(dist_util.dev())
    print("num of params: {} M".format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True,
        low_res=args.super_res,
        uncond_p=args.uncond_p,
        mode=args.mode,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        txt_file=args.txt_file,
        latent_root=args.latent_root,
        ms_feature_root=args.ms_feature_root,
    )

    logger.log("training...")
    TrainLoop(
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
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune_decoder = args.finetune_decoder,
        mode=args.mode,
        use_vgg=False,
        use_gan=False,
        use_tv=args.use_tv,
        uncond_p=args.uncond_p,
        super_res=args.super_res,
        use_tensorboard=args.use_tensorboard,
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
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="10",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder = False,
        mode="",
        use_tv=False,
        use_tensorboard=True,
        start_idx=-1,
        end_idx=-1,
        use_scale_shift_norm=True,
        txt_file="",
        predict_type="noise",
        latent_root='',
        ms_feature_root='',
        )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--ch_mult', dest='ch_mult', nargs='+', type=int)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
