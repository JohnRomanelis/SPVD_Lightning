from .shapenet_pointflow import ShapeNet15kPointClouds, synsetid_to_cate
from .utils import NoiseSchedulerDDPM

import torch
import numpy as np

from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from torch.utils.data import DataLoader
from torchsparse.utils.collate import sparse_collate_fn

def get_sid_to_int():
    ids = list(synsetid_to_cate.keys())
    ids.sort()
    sid_to_int = {k:i for i, k in enumerate(ids)}
    return sid_to_int

sid_to_int = get_sid_to_int()


class ShapeNet15kPointCloudsSparseNoisy(ShapeNet15kPointClouds):
    
    def set_voxel_size(self, voxel_size=1e-8):
        self.voxel_size = voxel_size
        
    def set_noise_params(self, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear'):
        self.noise_scheduler = NoiseSchedulerDDPM(beta_min, beta_max, n_steps, mode)
        
    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        
        pts = res['train_points']
        sid = res['sid']
        int_sid = sid_to_int[sid]

        pts, t, noise = self.noise_scheduler(pts)
        
        # turn to numpy to pass through sparse quantize 
        pts = pts.numpy()
        noise = noise.numpy()
        
        # Points will be store in a Point Tensor, that is a wrapper of the SparseTensor class
        # this means that the coordinates will be converted to a hashed representations
        coords = pts - np.min(pts, axis=0, keepdims=True)
        coords, indices = sparse_quantize(coords, self.voxel_size, return_index=True)
        
        # NOTE:
        # During training there is a highly small change of two points colliding with one another
        # however at this point this is not a problem 
        
        coords = torch.tensor(coords, dtype=torch.int)
        feats = torch.tensor(pts[indices], dtype=torch.float)
        noise = torch.tensor(noise[indices], dtype=torch.float)
        
        noisy_pts = SparseTensor(coords=coords, feats=feats)
        noise = SparseTensor(coords=coords, feats=noise)
        t = torch.tensor(t)
        
        return {'input':noisy_pts, 't':t, 'noise':noise, 'sid':int_sid}
    

def get_datasets(path, categories, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear', pres=1e-5):

    tr_dataset = ShapeNet15kPointCloudsSparseNoisy(
                categories=categories, split='train',
                tr_sample_size=2048,
                te_sample_size=2048,
                scale=1.,
                root_dir=path,
                normalize_per_shape=False,
                normalize_std_per_axis=False,
                random_subsample=True)

    te_dataset = ShapeNet15kPointCloudsSparseNoisy(
                categories=categories, split='val',
                tr_sample_size=2048,
                te_sample_size=2048,
                scale=1.,
                root_dir=path,
                normalize_per_shape=False,
                normalize_std_per_axis=False,
                random_subsample=True,
                all_points_mean=tr_dataset.all_points_mean,
                all_points_std=tr_dataset.all_points_std)

    tr_dataset.set_voxel_size(pres)
    tr_dataset.set_noise_params(beta_min=beta_min, beta_max=beta_max, n_steps=n_steps, mode=mode)

    te_dataset.set_voxel_size(pres)
    te_dataset.set_noise_params(beta_min=beta_min, beta_max=beta_max, n_steps=n_steps, mode=mode)

    return tr_dataset, te_dataset


def get_dataloaders(path, categories, beta_min=0.0001, beta_max=0.02, n_steps=1000, mode='linear', pres=1e-5, batch_size=32, num_workers=8):

    tr_dataset, te_dataset = get_datasets(path, categories, beta_min, beta_max, n_steps, mode, pres)

    tr_dl = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=sparse_collate_fn)
    te_dl = DataLoader(te_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=sparse_collate_fn)

    return tr_dl, te_dl