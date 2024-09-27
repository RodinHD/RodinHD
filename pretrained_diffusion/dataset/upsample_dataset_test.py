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
from functools import partial

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    train=True,
    low_res=0,
    uncond_p=0,
    mode='',
    txt_file='',
    start_idx=0,
    end_idx=10,
    scale=1.0,
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
        all_files = sorted([os.path.join(data_dir, x+'.npy') for x in all_files])[start_idx:end_idx]
    else:
        all_files = _list_image_files_recursively(data_dir)

    print(len(all_files))
    dataset = ImageDataset(
        image_size,
        all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        down_sample_img_size=low_res,
        uncond_p=uncond_p,
        mode = mode,
        scale=scale,
        latent_root=latent_root,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True, pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True
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
        elif os.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        down_sample_img_size=0,
        uncond_p=0,
        mode='',
        scale=1.0,
        latent_root='',
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.down_sample_img_size = down_sample_img_size
        self.uncond_p = uncond_p
        self.mode = mode
        self.resolution = resolution
        self.scale = scale
        self.latent_root = latent_root

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, 'rb') as f:
            triplane = np.load(f).astype(np.float32)
 
        triplane = th.as_tensor(triplane)
        if triplane.shape[-1] == 192:
            triplane  = th.stack([triplane[:,:,:64], triplane[:,:,64:128], triplane[:,:,128:]], 0) 

        triplane_lr = triplane.reshape(3,32,self.down_sample_img_size,self.down_sample_img_size)
        triplane_lr = th.cat([triplane_lr[0], triplane_lr[1], triplane_lr[2]], -1)
        
        path2_base = path.split('/')[-1].split('.')[0]

        parameters = th.load(os.path.join(self.latent_root, f"{path2_base}.pt"))
        mean, logvar = th.chunk(parameters, 2, dim=0)
        logvar = th.clamp(logvar, -30.0, 20.0)
        std = th.exp(0.5 * logvar)
        sample = th.rand_like(mean)
        label_tensor = mean + std * sample
        if random.random() < self.uncond_p:
            label_tensor = th.ones_like(label_tensor)

        data_dict = {"path": path, 'low_res': triplane_lr, 'ref': label_tensor}
 
        return triplane, data_dict
