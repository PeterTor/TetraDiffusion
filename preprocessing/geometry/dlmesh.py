# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

from render import mesh
from render import render
from render import regularizer

from geometry.dmtet import DMTet
###############################################################################
#  Geometry interface
###############################################################################

class DLMesh(torch.nn.Module):
    def __init__(self, deform, sdf, color, verts, indices, FLAGS,displacement_prior, rotation_matrices, scales):
        super(DLMesh, self).__init__()

        self.FLAGS = FLAGS
        self.grid_res = FLAGS.dmtet_grid
        self.verts = verts
        self.indices = indices

        self.marching_tets = DMTet(displacement_prior, rotation_matrices, scales)

        self.sdf = torch.nn.Parameter(sdf.clone(), requires_grad=False)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(deform.clone(), requires_grad=False)
        self.register_parameter('deform', self.deform)

        self.color = torch.nn.Parameter(color.clone(), requires_grad=True)
        self.register_parameter('color', self.color)

    @torch.no_grad()
    def getAABB(self):
        return mesh.aabb(self.mesh)

    def getMesh(self, material=None):
        v_deformed = self.verts #+ 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        v_colored = torch.clamp(self.color, 0, 1)
        # print(self.color.grad)
        v_feats = torch.cat([v_deformed, v_colored], dim=-1)
        assert v_feats.shape[-1] == (v_colored.shape[-1] + v_deformed.shape[-1])
        feats, faces = self.marching_tets(v_feats, self.sdf, self.deform, self.indices)
        verts = feats[..., :3]
        colors = feats[..., 3:6]
        imesh = mesh.Mesh(verts, faces, v_color=colors)
        imesh = mesh.auto_normals(imesh)
        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return opt_mesh, render.custom_render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'],
                                                   msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        opt_mesh, buffers = self.render(glctx, target, lgt, opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        #img_loss = img_loss + torch.nn.functional.l1_loss(buffers['depth'], target['depth'])
        #img_loss = img_loss + torch.nn.functional.l1_loss(buffers['mask'], target['mask'])

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer.
        if t_iter > 0.7:
            if self.FLAGS.laplace == "absolute":
                reg_loss += regularizer.laplace_regularizer_const(opt_mesh.v_pos, opt_mesh.t_pos_idx) * self.FLAGS.laplace_scale * t_iter #(1 - t_iter)
        # elif self.FLAGS.laplace == "relative":
        #     reg_loss += regularizer.laplace_regularizer_const(opt_mesh.v_pos - self.initial_guess.v_pos, opt_mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)

        return img_loss, reg_loss