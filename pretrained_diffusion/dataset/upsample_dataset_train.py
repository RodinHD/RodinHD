import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
import torch as th
from .degradation.bsrgan_light import degradation_bsrgan_variant as degradation_fn_bsr_light
from functools import partial
import json
from kornia import create_meshgrid

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    low_res_root=None,
    deterministic=False,
    train=True,
    low_res=0,
    uncond_p=0,
    num_patches=1,
    mode='',
    start_idx=-1,
    end_idx=-1,
    no_degradation=False,
    strong_degradation=False,
    txt_file='',
    pre_interpolate=True,
    image_root='',
    latent_root='',
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if txt_file != '':
        with open(txt_file) as f:
            all_files = f.read().splitlines()
        all_files = sorted([os.path.join(data_dir, x+'.npy') for x in all_files])
    else:
        all_files = _list_image_files_recursively(data_dir) 
    
    if start_idx >= 0 and end_idx >= 0 and start_idx < end_idx:
        all_files = all_files[start_idx:end_idx]
    print(len(all_files))

    dataset = ImageDataset(
        image_size,
        all_files,
        low_res_root=low_res_root,
        shard=MPI.COMM_WORLD.Get_rank() if train else 0,
        num_shards=MPI.COMM_WORLD.Get_size() if train else 1,
        down_sample_img_size=low_res,
        uncond_p=uncond_p,
        mode=mode,
        num_patches=num_patches,
        no_degradation=no_degradation,
        strong_degradation=strong_degradation,
        pre_interpolate=pre_interpolate,
        image_root=image_root,
        latent_root=latent_root,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=False
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        low_res_root=None,
        shard=0,
        num_shards=1,
        down_sample_img_size=0,
        uncond_p=0,
        mode='',
        no_degradation=False,
        num_patches=1,
        strong_degradation=False,
        pre_interpolate=True,
        image_root='',
        latent_root='',
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.down_sample_img_size = down_sample_img_size
        self.no_degradation = no_degradation
        self.low_res_root = low_res_root
        self.image_root = image_root
        self.latent_root = latent_root
        print("Use degradation: {} Strong: {} Low res root: {}".format(not self.no_degradation, strong_degradation, low_res_root))
 
        self.down_sample_img = partial(degradation_fn_bsr_light, sf=resolution//down_sample_img_size, strong=strong_degradation, skip_gaussian_noise=False) 
        self.uncond_p = uncond_p
        self.mode = mode
        self.resolution = resolution
        self.num_patches = num_patches
        self.pre_interpolate = pre_interpolate

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        triplane = np.load(path).astype(np.float32)
        triplane = triplane.reshape(96, self.resolution, self.resolution)

        triplane = triplane.transpose(1, 2, 0)
        triplane_lr = triplane
       
        if self.low_res_root is not None:
            low_res_root = random.choice(self.low_res_root)
            triplane_lr = np.load(os.path.join(low_res_root, path.split('/')[-1])).astype(np.float32)
            triplane_lr = triplane_lr.reshape(96, self.down_sample_img_size, self.down_sample_img_size).transpose(1,2,0)
        elif self.low_res_root is None and not self.no_degradation:
            triplane_lr = self.down_sample_img(triplane)["image"] 

        triplane = th.as_tensor(triplane).permute(2,0,1) 
        triplane_lr = th.as_tensor(triplane_lr).permute(2,0,1) 
        if self.low_res_root is None and self.no_degradation:
            triplane_lr = th.nn.functional.interpolate(triplane.unsqueeze(0), (self.down_sample_img_size, self.down_sample_img_size), mode='bicubic')[0]

        triplane = triplane.reshape(3,32,self.resolution,self.resolution)
        triplane = th.cat([triplane[0], triplane[1], triplane[2]], -1)

        if self.pre_interpolate:
            triplane_lr = th.nn.functional.interpolate(triplane_lr.unsqueeze(0), (self.resolution, self.resolution), mode='bicubic')[0]
            triplane_lr = triplane_lr.reshape(3,32,self.resolution,self.resolution)
        else:
            triplane_lr = triplane_lr.reshape(3,32,self.down_sample_img_size,self.down_sample_img_size)
        triplane_lr = th.cat([triplane_lr[0], triplane_lr[1], triplane_lr[2]], -1)

        path2_base = path.split('/')[-1].split('.')[0]
        feature = 0

        parameters = th.load(f"{self.latent_root}/{path2_base}.pt")
        mean, logvar = th.chunk(parameters, 2, dim=0)
        logvar = th.clamp(logvar, -30.0, 20.0)
        std = th.exp(0.5 * logvar)
        sample = th.rand_like(mean)
        label_tensor = mean + std * sample
        if random.random() < self.uncond_p:
            label_tensor = th.ones_like(label_tensor)

        fpath = os.path.join(self.image_root, path2_base)
        all_rgbs, all_rays = [], []
        
        r = np.random.rand()
        if r > 0.8:
            downsample = 2
        if r > 0.35 and r <= 0.8:
            downsample = 4
        if r <= 0.35:
            downsample = 8 
        
        for i in range(self.num_patches):
            f_idx = random.randint(0, 280) 

            rgbs, rays = read_meta(os.path.join(fpath, 'img_proc_fg_{:06d}.png'.format(f_idx)), os.path.join(fpath, 'metadata_{:06d}.json'.format(f_idx)), downsample=downsample)
            all_rgbs.append(rgbs)
            all_rays.append(rays)
        rgbs = all_rgbs
        rays = all_rays

        data_dict = {"ref":label_tensor, "path": path, 'low_res': triplane_lr, "rays": rays, "rgbs": rgbs, "vae_ms_feature": feature}

        return triplane, data_dict

def read_meta(img_path, camera_path, downsample=4):
    with open(camera_path, 'r') as f:
        meta = json.load(f)['cameras'][0]     
 
    w, h = int(meta['resolution'][0]/downsample), int(meta['resolution'][1]/downsample)
    img_wh = [w,h]
    focal_x =   meta['focal_length'] / meta['sensor_width']  * w
    focal_y =   meta['focal_length'] / meta['sensor_width']  * h
    cx, cy = 0.5*meta['resolution'][0]/downsample,  0.5*meta['resolution'][0]/downsample

    # ray directions for all pixels, same for all images (same H, W, focal)
    directions = get_ray_directions(h, w, [focal_x,focal_y], center=[cx, cy])  # (h, w, 3)
    directions = directions / th.norm(directions, dim=-1, keepdim=True)
    intrinsics = th.tensor([[focal_x,0,cx],[0,focal_y,cy],[0,0,1]]).float()        

    pose = np.array(meta['transformation']) 
    pose[:3, 3] = pose[:3, 3]/23.

    pose = nerf_matrix_to_ngp(pose, scale=0.6, offset=[0, 0, 0])

    c2w = th.FloatTensor(pose)

    image_path = os.path.join(img_path) 
    img = Image.open(image_path)
    if  downsample!=1.0:
        img = img.resize(img_wh, Image.LANCZOS)
    img = transforms.ToTensor()(img)  # (4, h, w)
    img = img.view(-1, w*h).permute(1, 0)  # (h*w, 4) RGBA
    if img.shape[-1]==4:
        img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
        
    rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
    all_rays = th.cat([rays_o, rays_d], 1)  # (h*w, 6)
     
    return img.reshape(1, h,w, 3), all_rays.reshape(1, h,w, 6)


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = th.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], th.ones_like(i)], -1)  # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose
