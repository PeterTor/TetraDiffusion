# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import os.path

import numpy as np
import torch
import trimesh

from pytorch3d.loss import chamfer_distance
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes

from render import mesh
from render import render
from render import regularizer

import kaolin
from kaolin.ops.conversions import marching_tetrahedra

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################


class DMTet:
    def __init__(self, grid_res=192):
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
            [4, 0, 3, -1, -1, -1],
            [1, 4, 2, 1, 3, 4],
            [3, 1, 5, -1, -1, -1],
            [2, 3, 0, 2, 5, 3],
            [1, 4, 0, 1, 5, 4],
            [4, 2, 5, -1, -1, -1],
            [4, 5, 2, -1, -1, -1],
            [4, 1, 0, 4, 5, 1],
            [3, 2, 0, 3, 5, 2],
            [1, 3, 5, -1, -1, -1],
            [4, 1, 2, 4, 3, 1],
            [3, 0, 4, -1, -1, -1],
            [2, 0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1]
        ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long,
                                                device='cuda')
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device='cuda')
        self.grid_res = grid_res

    ###############################################################################
    # Utility functions
    ###############################################################################
    def unique(self,x, dim=0):
        unique, inverse, counts = torch.unique(x, dim=dim,
                                               sorted=True, return_inverse=True, return_counts=True)
        decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
        inv_sorted = (inverse + decimals).argsort()
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        return unique, inverse, counts, index
    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)
        return torch.stack([a, b], -1)

    ###############################################################################
    # Marching tets implementation
    ###############################################################################
    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n >= 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device="cuda")
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 6)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1,
                         index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1] * 2,
            torch.stack((tet_gidx[num_triangles == 2] * 2, tet_gidx[num_triangles == 2] * 2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets * 2)

        return verts, faces, uvs, uv_idx


    def marching_cube_get_idx(self, sdf_n, tet_fx4):

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)

            v_id = torch.pow(2, torch.arange(4, dtype=torch.long))
            tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
            num_triangles = self.num_triangles_table[tetindex]

            tet_idx = torch.arange(tet_fx4.shape[0], device=sdf_n.device)[valid_tets]
            tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles == 2].unsqueeze(-1).expand(-1, 2).reshape(-1)),dim=0)
            return tet_idx

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx + 1) // 2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x, tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x, tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim=-1).view(-1, 3)

        return uvs, uv_idx


###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 0],
                                                                    (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[..., 1],
                                                                    (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, FLAGS, orig_mesh):
        super(DMTetGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.grid_res = grid_res

        tets = np.load('./tetra/{}_tets.npz'.format(grid_res))
        tet_verts = torch.tensor(tets['vertices'], dtype=torch.float, device="cpu")
        tet_faces = torch.tensor(tets['indices'], dtype=torch.long, device="cpu")
        self.verts = torch.tensor(tet_verts, dtype=torch.float32, device="cuda")

        if self.grid_res == 192:
            self.verts -= 0.5
        print(f"# of verts {len(self.verts)}")

        self.indices = torch.tensor(tet_faces, dtype=torch.long, device="cuda")
        self.generate_edges()

        self.marching_tets = DMTet(grid_res=self.grid_res)

        # Random init
        sdf = torch.rand_like(self.verts[:, 0]) - 0.1
        self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts[..., :3]), requires_grad=True)
        self.register_parameter('deform', self.deform)

        self.color = torch.nn.Parameter(torch.ones_like(self.verts) * 0.5, requires_grad=True)
        self.register_parameter('color', self.color)

        meshes = Meshes(verts=list(orig_mesh.v_pos[None, ...]),
                        faces=list(orig_mesh.t_pos_idx[None, ...]))
        vs_watertight, fs_watertight = self.get_convex(meshes.verts_packed(), meshes.faces_packed())
        self.mask = self.kaolin_mesh_to_mask(vs_watertight[None, ...], fs_watertight, self.verts[None, ...])
        self.deform_scale = 0.45
        self.with_laplace = False

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device="cuda")
            all_edges = self.indices[:, edges].reshape(-1, 2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.test_me = all_edges_sorted
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def clamp_deform(self):
        self.sdf.data[:] = self.sdf.data.clamp(-1.0, 1.0)
        self.deform.data[:] = self.deform.data.clamp(-1.0, 1.0)
        self.color.data[:] = self.color.data.clamp(0.0, 1.0)

    def getMesh(self, material=None, mask=False):
        v_deformed = self.verts + 2 / (self.grid_res * 2) * self.deform * self.deform_scale
        sdf = self.sdf
        v_colored = self.color
        if mask:
            with torch.no_grad():
                sdf[self.mask] = 1.0
        v_feats = torch.cat([v_deformed, v_colored], dim=-1)
        feats, faces, uvs, uv_idx = self.marching_tets(v_feats, sdf, self.indices)
        verts = feats[..., :3]
        colors = feats[..., 3:]
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material, v_color=colors)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        results = render.custom_render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt,target['resolution'], spp=target['spp'],msaa=True, background=target['background'], bsdf=bsdf)
        return opt_mesh,results

    def get_convex(self, vs, fs):
        mesh = trimesh.Trimesh(vs.detach().cpu().double().numpy()*1.07, fs.detach().cpu().numpy().astype(np.int32),repair=False).convex_hull
        vs = torch.tensor(mesh.vertices).float().cuda()
        fs = torch.tensor(mesh.faces).long().cuda()
        return vs, fs

    def kaolin_mesh_to_mask(self, verts_bxnx3, face_fx3, points_bxnx3):
        verts_bxnx3 = verts_bxnx3.cuda()
        face_fx3 = face_fx3.cuda()
        points_bxnx3 = points_bxnx3.cuda()

        sign = ~kaolin.ops.mesh.check_sign(verts_bxnx3, face_fx3, points_bxnx3, hash_resolution=1024) # (0: inside; 1: outside)
        return sign.squeeze(0)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        opt_mesh, buffers = self.render(glctx, target, lgt, opt_material)

        t_iter = iteration / self.FLAGS.iter

        color_ref = target['img']

        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:],color_ref[..., 0:3] * color_ref[..., 3:])

        color_ref2 = target['img_second']
        img_loss = 10. * img_loss + 2 * torch.nn.functional.mse_loss(buffers['shaded_second'][..., 3:],color_ref2[..., 3:])
        img_loss = img_loss + 2 * loss_fn(buffers['shaded_second'][..., 0:3] * color_ref2[..., 3:],color_ref2[..., 0:3] * color_ref2[..., 3:])

        if iteration % 300 == 0 and iteration < 1790 and not self.with_laplace:
            self.deform.data[:] *= 0.4

        sdf_weight = (self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.5) * min(1, 4.0 * t_iter))
        reg_loss = 10 * sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight  # Dropoff to 0.01

        meshes = Meshes(verts=list(opt_mesh.v_pos[None, ...]), faces=list(opt_mesh.t_pos_idx[None, ...]))
        pred_points, pred_normals = sample_points_from_meshes(meshes, num_samples=20000, return_normals=True,return_textures=False)

        chamfer, _ = chamfer_distance(x=target['points'], y=pred_points, single_directional=False)
        normals = torch.nn.functional.l1_loss(buffers['normal'][..., :3], target['normal'][..., :3])
        geo_normals = torch.nn.functional.l1_loss(buffers['geo_normal'][..., :3], target['geo_normal'][..., :3])
        normals_second = torch.nn.functional.l1_loss(buffers['normal_second'][..., :3], target['normal_second'][..., :3])
        geo_normals_second = torch.nn.functional.l1_loss(buffers['geo_normal_second'][..., :3], target['geo_normal_second'][..., :3])

        img_loss = img_loss + (10 * normals + 10 * geo_normals + 0.1 * normals_second + 0.1 * geo_normals_second)
        img_loss = img_loss + sdf_weight * 100. * torch.nn.functional.l1_loss(buffers['depth'], target['depth'])
        img_loss = img_loss + sdf_weight * 50. * torch.nn.functional.l1_loss(buffers['depth_second'], target['depth_second'])

        img_loss += chamfer

        if self.with_laplace:
            reg_loss += 50 * regularizer.laplace_regularizer_const(opt_mesh.v_pos, opt_mesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter)

        return img_loss, reg_loss

