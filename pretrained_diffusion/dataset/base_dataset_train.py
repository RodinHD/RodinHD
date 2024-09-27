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
    start_idx=-1,
    end_idx=-1,
    txt_file='',
    latent_root=None,
    ms_feature_root=None,
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
        shard=MPI.COMM_WORLD.Get_rank() if train else 0,
        num_shards=MPI.COMM_WORLD.Get_size() if train else 1,
        down_sample_img_size=low_res,
        uncond_p=uncond_p,
        mode=mode,
        latent_root=latent_root,
        ms_feature_root=ms_feature_root,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True
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
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        down_sample_img_size=0,
        uncond_p=0,
        mode='',
        latent_root=None,
        ms_feature_root=None,
    ):
        super().__init__()
        self.local_images = image_paths[shard:][::num_shards]
        self.down_sample_img_size = down_sample_img_size
        self.uncond_p = uncond_p
        self.mode = mode
        self.resolution = resolution
        self.latent_root = latent_root
        self.ms_feature_root = ms_feature_root

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, 'rb') as f:
            triplane = np.load(f).astype(np.float32)
 
        triplane = th.as_tensor(triplane)
        triplane = triplane.reshape(3,32,self.resolution,self.resolution)
        triplane = th.cat([triplane[0], triplane[1], triplane[2]], -1)

        path2_base = path.split('/')[-1].split('.')[0]
        
        parameters = th.load(os.path.join(self.latent_root, f"{path2_base}.pt"))
        mean, logvar = th.chunk(parameters, 2, dim=0)
        logvar = th.clamp(logvar, -30.0, 20.0)
        std = th.exp(0.5 * logvar)
        sample = th.rand_like(mean)
        label_tensor = mean + std * sample
        feature = [feat.to(th.float32) for feat in th.load(os.path.join(self.ms_feature_root, f"{path2_base}.pt"))]
        if random.random() < self.uncond_p:
            label_tensor = th.ones_like(label_tensor)
            feature = [th.ones_like(feat) for feat in feature]
        data_dict = {"ref": label_tensor, "path": path, "vae_ms_feature": feature}
        return triplane, data_dict

