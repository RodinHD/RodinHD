import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import multiprocessing as mp  
import concurrent.futures  

import trimesh
import math

import torch
from torch.utils.data import DataLoader

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def spiral(radius=1):
    return lambda theta, phi : [
                                radius*np.sin(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))) * np.sin(math.pi-theta), # 1 
                                radius*np.cos(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))), # 2 
                                radius * np.sin(np.arccos(1 - 2*(np.clip(phi, 1e-5, math.pi - 1e-5)/ math.pi))) * np.cos(math.pi-theta), # 3
                                ]


def gen_path_spiral(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180, drange=360.0):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(-3.14/2 + 0.35 * np.sin(2 * 3.14 * t / frames), 3.14/2 + 0.3 * np.cos(2 * 3.14 * t / frames)))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)


def circle(radius=3.5, h=0.0, axis='z', t0=0, r=1):
    if axis == 'z':
        # return lambda t: [radius * np.cos(r * t + t0), radius * np.sin(r * t + t0), h]
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)][::-1]
    elif axis == 'y':
        return lambda t: [radius * np.cos(r * t + t0), h, radius * np.sin(r * t + t0)]
    else:
        return lambda t: [h, radius * np.cos(r * t + t0), radius * np.sin(r * t + t0)]


def gen_path(pos_gen, at=(0, 0, 0), up=(0, -1, 0), frames=180, drange=360.0):
    c2ws = []
    for t in range(frames):
        c2w = torch.eye(4)
        cam_pos = torch.tensor(pos_gen(t * (drange / frames) / 180 * np.pi))
        cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
        c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
        c2ws.append(c2w)
    return torch.stack(c2ws)


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = torch.zeros_like(camera_position)
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(z_axis, up, -1))[0]
    y_axis = normalize(cross(x_axis, z_axis, -1))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2 == 0] = 1
        return x / l2,


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def circle_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    at = (0.,0.,0.6)
    up = up
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.ones(size, device=device) * 90 * np.pi / 180
    phi_step = 2*np.pi / size
    phis = torch.tensor([i*phi_step for i in range(size)], device=device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, root_path, save_dir,  device, num_train_frames=300, type='train', downscale=1, n_test=10, triplane_resolution=512, triplane_channels=32):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.subject_id = root_path
        self.subject_id = root_path.split("/")[-1]
        self.root_path = root_path
        self.save_dir = save_dir
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.triplane_resolution = triplane_resolution
        self.triplane_channels = triplane_channels

        if self.type == 'test_video':
            scene_bbox = torch.tensor([[-self.bound, -self.bound, -self.bound], [self.bound, self.bound, self.bound]])
            center = torch.mean(scene_bbox, dim=0)
            radius = torch.norm(scene_bbox[1]-center)*1.45
            up = [0, -1 ,0]

            # pos_gen = circle(radius=radius, h=-0.2, axis='z', t0=80) #105
            # self.poses = gen_path(pos_gen, at=(0., 0.4, 0.), up=up, frames=60)
            pos_gen = spiral(radius=radius)
            self.poses = gen_path_spiral(pos_gen, at=(0.,0.4,0.), up=up,frames=60)
            self.images = None

            with open(os.path.join(self.root_path,  'metadata_000000.json'), 'r') as f:
                self.meta = json.load(f)['cameras'][0] 
    
            w, h = int(self.meta['resolution'][0]/self.downscale), int(self.meta['resolution'][1]/self.downscale)
            self.focal_x =   self.meta['focal_length'] / self.meta['sensor_width']  * w
            self.focal_y =   self.meta['focal_length'] / self.meta['sensor_width']  * h
            self.W = w
            self.H = h
        else:
 
            with open(os.path.join(self.root_path,  'metadata_000000.json'), 'r') as f:
                self.meta = json.load(f)['cameras'][0] 
    
            w, h = int(self.meta['resolution'][0]/self.downscale), int(self.meta['resolution'][1]/self.downscale)
            self.focal_x =   self.meta['focal_length'] / self.meta['sensor_width']  * w
            self.focal_y =   self.meta['focal_length'] / self.meta['sensor_width']  * h
            self.W = w
            self.H = h

            frames = list(range(0,  num_train_frames)) if (self.training or self.type == 'test_all') else list(range(0,  5))
  
            def load_data(i):  
                camera_path = os.path.join(self.root_path,   'metadata_{:06d}.json'.format(i))  
                with open(camera_path, 'r') as f:  
                    camera_ = json.load(f)['cameras'][0]  
            
                pose = np.array(camera_['transformation'], dtype=np.float32) # assume [4, 4]  
                pose[:3, 3] = pose[:3, 3]/23.  
            
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)  
            
                image_path = os.path.join(self.root_path,  'img_proc_fg_{:06d}.png'.format(i))  
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]  
            
                # add support for the alpha channel as a mask.  
                if image.shape[-1] == 3:   
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
                else:  
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)  
            
                if image.shape[0] != self.H or image.shape[1] != self.W:  
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)  
            
                image = image.astype(np.float32) / 255 # [H, W, 3/4]  
            
                return pose, image  
            
            # frames = list(range(0,  num_train_frames)) if (self.training or self.type == 'test_all') else list(range(0,  200))  
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(mp.cpu_count(), 16)) as executor:  
                results = list(executor.map(load_data, frames))  
            
            self.poses, self.images = zip(*results)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

 
        cx =   (self.W / 2)
        cy =   (self.H / 2)
    
        self.intrinsics = np.array([self.focal_x, self.focal_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]
 
        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None

        triplane_path = os.path.join(self.save_dir, self.subject_id+'.npy')
        if os.path.exists(triplane_path):
            print("Loading triplane from {}".format(triplane_path))
            with open(triplane_path, 'rb') as f:
                triplane = np.load(f)
                triplane = torch.as_tensor(triplane, dtype=torch.float32)
                
        else:
            triplane = 0.1*torch.randn((3, self.triplane_channels, self.triplane_resolution, self.triplane_resolution))
       
        if self.training:
            iwc_state_path = os.path.join(self.save_dir, self.subject_id+'_iwc_state.ckpt')
            if os.path.exists(iwc_state_path):
                iwc_state = torch.load(iwc_state_path, map_location='cpu')
            else:
                iwc_state = {
                    "fisher_state": [],
                    "optpar_state": [],
                }
            return loader, triplane.to(torch.float32), iwc_state
        else:
            return loader, triplane.to(torch.float32)
