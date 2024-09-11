import models.ddpm_unet as ddpm_unet
import models.ddpm_unet_attn as ddpm_unet_attn
import models.spvd as spvd
from functools import partial

from models.lightningBase import DiffusionBase


__all__ = ['DiffusionBase', 'SPVD_S', 'SPVD', 'SPVD_L']

# SPVD-S : 
SPVD_S = partial(ddpm_unet_attn.SPVUnet, voxel_size=0.1, nfs=(32, 64, 128, 256), num_layers=1, attn_chans=8, attn_start=3)

# SPVD : 32.9M parameters
SPVD = partial(spvd.SPVUnet, point_channels=3, voxel_size=0.1, num_layers=1, pres=1e-5,
                    down_blocks = [[(32, 64, 128, 192, 192, 256), 
                                    (True, True, True, True, False), 
                                    (None, None, None, 8, 8)]], 
                                    # BLOCK 1
                    up_blocks   = [[(256, 192, 192), 
                                    (True, True), 
                                    (8, 8), 
                                    (3, 3)], 
                                    # BLOCK 2
                                   [(192, 128, 64, 32), 
                                    (True, True, False), 
                                    (None, None, None), 
                                    (3, 3, 3)]])

# SPVD-L : 88.1M parameters                 
SPVD_L = partial(spvd.SPVUnet, point_channels=3, voxel_size=0.1, num_layers=1, pres=1e-5,
                    down_blocks = [[(64, 128, 192, 256, 384, 384), 
                                    (True, True, True, True, False), 
                                    (None, None, None, 8, 8)]], 
                                    # BLOCK 1
                    up_blocks   = [[(384, 384, 256), 
                                    (True, True), 
                                    (8, 8), 
                                    (3, 3)], 
                                    # BLOCK 2
                                   [(256, 192, 128, 64), 
                                    (True, True, False), 
                                    (None, None, None), 
                                    (3, 3, 3)]])