import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from .glide_util import sample
from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .losses import  TVLoss
from .adv import AdversarialLoss
from .resample import LossAwareSampler, UniformSampler
from .utils import checkpoint
import glob
import torchvision.utils as tvu
import PIL.Image as Image
import imageio
from .lpips.lpips import LPIPS
from .visualizer import Visualizer

from collections import OrderedDict
import deepspeed
from torchvision.transforms.functional import normalize
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
 
 
def render_fn(renderer, triplane, rays_o, rays_d, max_ray_batch, bg_color, perturb, max_steps):  
    return renderer.render(triplane, rays_o, rays_d, staged=True, max_ray_batch=max_ray_batch, bg_color=bg_color, perturb=perturb, max_steps=max_steps)  

class TrainLoop:
    def __init__(
        self,
        args,
        model,
        glide_options,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        finetune_decoder=False,
        mode='',
        use_vgg=False,
        use_gan=False,
        use_tv=False,
        use_renderer="",
        uncond_p=0,
        super_res=0,
        image_size=256,
        use_tensorboard=True,
        render_weight=1.0,
        render_lpips_weight=0.5,
        patch_size=64,
        scale=1.0,
        loss_type=None,
        predict_type='xstart',
    ):
        self.model = model
        self.image_size = image_size
        self.glide_options = glide_options
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = find_resume_checkpoint(resume_checkpoint)
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.schedule_sampler =schedule_sampler
        self.render_weight = render_weight
        self.render_lpips_weight = render_lpips_weight
        self.patch_size = patch_size
        self.scale = scale
        self.predict_type = predict_type
        print("Training with predict type: ", self.predict_type)

        if use_renderer:
            from  Renderer.TriplaneFit.network import NeRFNetwork
            self.renderer = NeRFNetwork(
                resolution=[image_size] * 3,
                sigma_rank=[8, 8, 8],
                color_rank=[24, 24, 24],
                bg_resolution=[512, 512],
                bg_rank=8,
                color_feat_dim=24,
                num_layers=3,
                hidden_dim=128,
                num_layers_bg=2,
                hidden_dim_bg=64,
                bound=1.0,
                cuda_ray=True,
                density_scale=1,
                min_near=0.2,
                density_thresh=10,
                bg_radius=-1,
                grid_size=128,
            )         
            ckpt = th.load(use_renderer, map_location="cpu")
        
            loading_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                if not 'density_grid' in k and not 'density_bitfield' in k:
                    loading_dict[k] = v
                    print(k)
            self.renderer.load_state_dict(loading_dict, strict=False)
            self.renderer.to(dist_util.dev())
        else:
            self.renderer = None

        if use_vgg:
            self.vgg = LPIPS(net_type="vgg").to(dist_util.dev()).eval()
            print(f'use perc vgg')
        else:
            self.vgg = None

        if use_gan:
            self.adv = AdversarialLoss()
            print('use adv')
        else:
            self.adv = None

        if use_tv:
            self.tvreg =  TVLoss()
            print('use TV')
        else:
            self.tvreg = None
        
        if loss_type == 'l1':
            self.loss = th.nn.L1Loss()
        else:
            self.loss = th.nn.MSELoss()
            
        self.super_res = super_res
         
        self.uncond_p =uncond_p
        self.mode = mode

        self.finetune_decoder = finetune_decoder
        if finetune_decoder:
            self.optimize_model = self.model
        else:          
            self.optimize_model = self.model.encoder
         
        self.model_params = list(self.optimize_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        if self.use_fp16:
            self._setup_fp16()

        self.model_engine, self.opt, _, _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=self.master_params)
        self._load_and_sync_parameters()
        if self.resume_step:
            # self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard and dist.get_rank() == 0:
            self.writer = Visualizer(os.path.join(logger.get_dir(), 'tf_events'))

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            print("resume checkpoint: ", resume_checkpoint)
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            print("resume step: ", self.resume_step)
            self.model_engine.load_checkpoint(get_blob_logdir(), f"model{(self.resume_step):06d}")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step   <= self.lr_anneal_steps
        ):
            batch, model_kwargs = next(self.data)

            self.run_step(batch, model_kwargs)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
         
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, model_kwargs):
        self.forward_backward(batch, model_kwargs)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, model_kwargs):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())

            micro_cond={n:model_kwargs[n][i:i+self.microbatch].to(dist_util.dev()) for n in model_kwargs if n  in ['ref', 'low_res']}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            with th.no_grad():
                model_kwargs["ref"] = model_kwargs["ref"].to(dist_util.dev())

            micro = micro * self.scale
            if self.predict_type == 'noise':
                losses = self.diffusion.training_losses(
                    self.model_engine,
                    micro,
                    t,
                    vgg=None,
                    adv=None,
                    tvreg=None,
                    model_kwargs={'ref': micro_cond["ref"], 'low_res': micro_cond['low_res'] * self.scale}
                )
                loss = (losses["loss"] * weights).mean()
            else:
                # predict xstart or v
                upsampled_triplane, loss_weight, target =  self.diffusion.training_forward(
                        self.model_engine,
                        micro,
                        t,
                        model_kwargs={'ref': micro_cond["ref"], 'low_res': micro_cond['low_res'] * self.scale})

                loss_rec = self.loss(upsampled_triplane, target) * loss_weight
                upsampled_triplane = upsampled_triplane / self.scale
                loss = 0
                loss += loss_rec 

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, loss.detach()
                )
                loss = loss * weights

            if self.renderer is not None:
                triplane = th.cat([upsampled_triplane[:, :, :, :self.image_size], upsampled_triplane[:, :, :, self.image_size:self.image_size*2], upsampled_triplane[:, :, :, self.image_size*2:]], 0)
                triplane = triplane.reshape(3, 1, 32, self.image_size, self.image_size)
                self.renderer.reset_extra_state()
                for _ in range(16):
                    self.renderer.update_extra_state(triplane) 
                allrays_list, allrgbs_list  = model_kwargs['rays'], model_kwargs['rgbs']
                for patch_idx in range(len(allrays_list)):
                    allrays, allrgbs  = allrays_list[patch_idx], allrgbs_list[patch_idx]
                    crop_size = self.patch_size
                    ray_idx = 0
                    crop_h = np.random.randint(0, allrgbs.shape[2]-crop_size) if allrgbs.shape[2] > crop_size else 0
                    crop_w = np.random.randint(0, allrgbs.shape[2]-crop_size) if allrgbs.shape[2] > crop_size else 0
                    rays_train, rgb_train = allrays[:,ray_idx, crop_h:crop_h+crop_size, crop_w:crop_w+crop_size], allrgbs[:,ray_idx, crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]
                    if rgb_train.sum() > th.ones_like(rgb_train).sum() * 0.7:
                        # Reject white patch
                        if allrgbs.shape[2] == crop_size:
                            crop_h, crop_w = 0, 0
                        else:
                            crop_h = np.random.randint(0, allrgbs.shape[2]-crop_size)
                            crop_w = np.random.randint(0, allrgbs.shape[2]-crop_size)
                        rays_train, rgb_train = allrays[:,ray_idx, crop_h:crop_h+crop_size, crop_w:crop_w+crop_size], allrgbs[:,ray_idx, crop_h:crop_h+crop_size, crop_w:crop_w+crop_size]
                    rays, rgbs = rays_train.reshape(-1,crop_size*crop_size ,6), rgb_train.reshape(-1,crop_size*crop_size ,3)
                    rays = rays.to(dist_util.dev())
                    rgbs = rgbs.to(dist_util.dev())
    
                    rays_o = rays[:, :, :3] # [B, N, 3]
                    rays_d = rays[:, :, 3:] # [B, N, 3]

                    output_image = None
                    if rays.shape[1] > 4096:
                        for i in range(0, rays.shape[1], 4096):
                            # Use checkpoint to avoid OOM
                            outputs = checkpoint(render_fn, (self.renderer, triplane, rays_o[:, i:min(i+4096, rays.shape[1])], rays_d[:, i:min(i+4096, rays.shape[1])], 4096, None, False, 1024), None, True)
                            output_image = outputs['image'] if output_image is None else th.cat([output_image, outputs['image']], 1)
                    else:
                        outputs = self.renderer.render(triplane, rays_o, rays_d, staged=True, max_ray_batch=4096, bg_color=None, perturb=False, max_steps=1024)
                        output_image = outputs['image']
                
                    gt_img = rgbs.reshape(1,crop_size,crop_size,3).permute(0,3,1,2)
                    output_img = output_image.clamp(0.0, 1.0).reshape(1,crop_size,crop_size,3).permute(0,3,1,2)

                    loss_render =  th.nn.L1Loss()(gt_img, output_img)
                    loss += (loss_render*self.render_weight/len(allrays_list)) if 'face_rgbs' not in model_kwargs else (loss_render*self.render_weight / (len(allrays_list) + 1))

                    if  self.vgg is not None and self.renderer is not None:
                        loss_lpips = self.vgg(output_img*2 -1., gt_img*2 - 1.)*self.render_lpips_weight
                        loss += (loss_lpips / len(allrays_list)) if 'face_rgbs' not in model_kwargs else (loss_lpips / (len(allrays_list) + 1))
                    else:
                        loss_lpips = th.tensor(0.)

                    if self.step % 100 == 0 and dist.get_rank() == 0:
                        s_path = os.path.join(logger.get_dir(), 'train_images')
                        os.makedirs(s_path,exist_ok=True)
                        output_image = output_image.clamp(0.0, 1.0)
                        rgb_map  = output_image.reshape(crop_size, crop_size, 3).cpu() 
                        rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                        imageio.imwrite(os.path.join(s_path, "render_iter_{:08}_t_{:02}_patch_{:02}.png".format(self.step, int(t), patch_idx)), rgb_map)

                        rgb_map  = rgbs.reshape(crop_size, crop_size, 3).cpu() 
                        rgb_map = (rgb_map.detach().numpy() * 255).astype('uint8')
                        imageio.imwrite(os.path.join(s_path, "gt_iter_{:08}_t_{:02}_patch_{:02}.png".format(self.step, int(t), patch_idx)), rgb_map)
            else:
                loss_render = th.tensor(0.)
                loss_lpips = th.tensor(0.)

            if  self.adv is not None and self.renderer is not None:
                loss_adv = self.adv(output_img*2 -1., gt_img*2 - 1.)*0.01
                loss += loss_adv[0]
            else:
                loss_adv = th.tensor(0.)

            if self.tvreg is not None:
                loss_l1 = th.mean(th.abs(upsampled_triplane)) * 1e-4
                loss_tv = 0
                for i in range(len(triplane)):
                    loss_tv = loss_tv + self.tvreg(triplane[i]) * 2e-4  
                loss_dist = self.renderer.dist_loss(triplane) * 2e-5
                loss += loss_l1
                loss += loss_tv
                loss += loss_dist
            else:
                loss_tv = th.tensor(0.)
                loss_l1 = th.tensor(0.)
                loss_dist = th.tensor(0.)

            if self.predict_type != 'noise':
                log_loss_dict(
                    {'rec': loss_rec, 'pixel': loss_render, 'lpips': loss_lpips, 'adv': loss_adv, 'loss_tv': loss_tv, 'loss_l1': loss_l1, 'loss_dist': loss_dist, 'loss_weight': th.tensor(loss_weight).clone().detach().to(th.float32)},
                )
                if self.use_tensorboard and self.step % self.log_interval == 0 and dist.get_rank() == 0:
                    self.writer.write_dict({'Train/rec': loss_rec, 'Train/pixel': loss_render, 'Train/lpips': loss_lpips, 'Train/adv': loss_adv, 'Train/loss_tv': loss_tv, 'Train/loss_l1': loss_l1, 'Train/loss_weight': th.tensor(loss_weight).clone().detach()}, self.step)
            else:
                log_loss_dict_predict_noise(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )
                if self.use_tensorboard and self.step % self.log_interval == 0 and dist.get_rank() == 0:
                    self.writer.write_dict({"loss": loss.item()}, self.step)

            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                self.model_engine.backward(loss)

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self.model_engine.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        for name, param in self.optimize_model.named_parameters():
            if param.grad == None:
                print(name, )

        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        return
   
    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                th.save(state_dict, os.path.join(get_blob_logdir(), filename))
  
        self.model_engine.save_checkpoint(get_blob_logdir(), f"model{(self.step+self.resume_step):06d}")

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            th.save(self.opt.state_dict(), os.path.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"))
 
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.optimize_model.parameters()), master_params
            )
        state_dict = self.optimize_model.state_dict()
        for i, (name, _value) in enumerate(self.optimize_model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.optimize_model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    filename=filename.split('/')[-1]
    assert(filename.endswith(".pt"))
    filename=filename[:-3]
    if filename.startswith("model"):
        split = filename[5:]
    elif filename.startswith("ema"):
        split = filename.split("_")[-1]
    else:
        return 0
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    p=os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(p,exist_ok=True)
    return p

def find_resume_checkpoint(resume_checkpoint):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if not resume_checkpoint:
        return None
    if "ROOT" in resume_checkpoint:
        maybe_root=os.environ.get("AMLT_MAP_INPUT_DIR")
        maybe_root="OUTPUT/log" if not maybe_root else maybe_root
        root=os.path.join(maybe_root,"checkpoints")
        resume_checkpoint=resume_checkpoint.replace("ROOT",root)
    if "LATEST" in resume_checkpoint:
        files=glob.glob(resume_checkpoint.replace("LATEST","*.pt"))
        if not files:
            return None
        return max(files,key=parse_resume_step_from_filename)
    return resume_checkpoint



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict( losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
    

def log_loss_dict_predict_noise(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
