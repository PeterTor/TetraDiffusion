# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes

from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset
from glob import glob

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################
from pytorch3d.transforms import RotateAxisAngle

class DatasetMesh(Dataset):

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Load environment map texture
        print("Loading light")
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        print("Computing tangents")
        self.ref_mesh = mesh.compute_tangents(ref_mesh) if ref_mesh.v_tex is not None else ref_mesh
        print("Done with ref mesh")


    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (((itr / 100) % 4) * 180) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        #self.envlight = light.load_env(self.irrmaps[np.random.randint(0,len(self.irrmaps))], scale=self.env_scale)
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        # if torch.rand([1]) > 0.9:
        #     cam_radius = 0.3  # torch.rand([1]) * 0.2 + 1.1
        # else:
        #     cam_radius = self.cam_radius  # torch.rand([1]) * 0.2 + 0.2
        # cam_radius = torch.clamp(cam_radius[0], 0.6, self.cam_radius)
        mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.25)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, iter_res, iter_spp = self._rotate_scene(itr)
        else:
            mv, mvp, campos, iter_res, iter_spp = self._random_scene()
        ref_mesh = self.ref_mesh
        # angle = np.random.randint(-180,180)
        # rot_z_90 = RotateAxisAngle( angle, axis="Y", degrees=True).to("cuda")
        # ref_mesh.v_pos = rot_z_90.transform_points(ref_mesh.v_pos)
        # if itr < 500:
        #     self.envlight
        rendered = render.render_mesh(self.glctx,ref_mesh , mvp, campos, self.envlight, iter_res, spp=iter_spp,num_layers=self.FLAGS.layers, msaa=True, background=None)
        img = rendered['shaded']
        img_second = rendered['shaded_second']
        # img_third = rendered['shaded_third']
        depth = rendered['depth']
        depth_second = rendered['depth_second']
        # depth_third = rendered['depth_third']
        mask = rendered['mask']
        mask_second = rendered['mask_second']
        normal = rendered['normal']
        normal_second = rendered['normal_second']
        geo_normal = rendered['geo_normal']
        geo_normal_second = rendered['geo_normal_second']

        meshes = Meshes(verts=list(self.ref_mesh.v_pos[None, ...]), faces=list(self.ref_mesh.t_pos_idx[None, ...]))
        # gt_points = sample_points(self.ref_mesh.v_pos[None, ...], self.ref_mesh.t_pos_idx, 200000)[0]
        gt_points, gt_normals = sample_points_from_meshes(meshes, num_samples=20000, return_normals=True, return_textures=False)

        return {
            'mv': mv,
            'mvp': mvp,
            'campos': campos,
            'resolution': iter_res,
            'spp': iter_spp,
            'img': img,
            'depth': depth,
            'mask': mask,
            'points': gt_points,
            'points_normals': gt_normals,
            'normal': normal,
            'geo_normal': geo_normal,
            'orig_mesh': ref_mesh,
            'img_second': img_second,
            'depth_second': depth_second,
            'mask_second': mask_second,
            'normal_second': normal_second,
            'geo_normal_second': geo_normal_second,
            # 'img_third': img_third,
            # 'depth_third': depth_third
        }
