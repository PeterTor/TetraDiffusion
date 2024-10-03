# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']

        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'depth' : torch.cat(list([item['depth'] for item in batch]), dim=0),
            'mask' : torch.cat(list([item['mask'] for item in batch]), dim=0),
            'normal' : torch.cat(list([item['normal'] for item in batch]), dim=0),
            'geo_normal' : torch.cat(list([item['geo_normal'] for item in batch]), dim=0),
            'img_second' : torch.cat(list([item['img_second'] for item in batch]), dim=0),
            # 'img_third' : torch.cat(list([item['img_third'] for item in batch]), dim=0),
            'depth_second' : torch.cat(list([item['depth_second'] for item in batch]), dim=0),
            # 'depth_third' : torch.cat(list([item['depth_third'] for item in batch]), dim=0),
            'mask_second' : torch.cat(list([item['mask_second'] for item in batch]), dim=0),
            'normal_second' : torch.cat(list([item['normal_second'] for item in batch]), dim=0),
            'geo_normal_second' : torch.cat(list([item['geo_normal_second'] for item in batch]), dim=0),
            'points' : batch[0]['points'],
            'points_normals' : batch[0]['points_normals'],
            'orig_mesh' : batch[0]['orig_mesh']
        }