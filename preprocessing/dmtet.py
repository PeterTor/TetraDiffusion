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
from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import sample_points_from_meshes

from render import mesh
from render import render
from render import regularizer


###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self, displacement_prior, rotations, scales, grid_res=256):
        self.displacement_prior = displacement_prior
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
        # self.v_any = torch.randn((1, 3), device='cuda')

        self.rotations = rotations
        self.scales = scales
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

    def __call__(self, feats_nxc, sdf_n, dir_nx2, tet_fx4, return_tet_idx=False):
        c_dim = feats_nxc.shape[1]
        sdf_n = torch.tanh(sdf_n)

        with torch.no_grad():
            occ_n = sdf_n > 0
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


        # Check x**2 + y**2 < 4*r**2
        dir_nx2 = 4./self.grid_res * torch.tanh(dir_nx2)
        # norms = torch.linalg.vector_norm(dir_nx2, ord=2, dim=1)
        # mask = norms > 2.5*self.displacement_prior.squeeze()
        # clipper = torch.ones_like(self.displacement_prior)
        # clipper[mask] = 2.5*self.displacement_prior[mask, :] / norms[mask].unsqueeze(1)
        # dir_nx2 = dir_nx2 * clipper

        rotations_to_interp = self.rotations[interp_v.reshape(-1)]  # .reshape(-1, 2, 3, 3)
        scales_to_interp = self.scales[interp_v.reshape(-1)][..., None]
        edges_to_interp = feats_nxc[interp_v.reshape(-1)].reshape(-1, 2, c_dim)
        lambda_to_interp = dir_nx2[interp_v.reshape(-1)].reshape(-1, 2)

        axis_aligned_vectors = torch.zeros_like(rotations_to_interp[:, :, 0]).squeeze()
        axis_aligned_vectors[..., 2] = 1.0
        axis_aligned_vectors[:, :2] = lambda_to_interp
        rotated_vectors = torch.einsum("bi,bji->bj", axis_aligned_vectors, torch.linalg.inv(rotations_to_interp)) * scales_to_interp
        rotated_vectors = rotated_vectors.reshape(-1, 2, 3)

        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1
        denominator = edges_to_interp_sdf.sum(1, keepdim=True)
        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        edges_to_interp[..., :3] = rotated_vectors

        vert_feats = (edges_to_interp * edges_to_interp_sdf).sum(1)

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
        if return_tet_idx:
            tet_idx = torch.arange(tet_fx4.shape[0], device=feats_nxc.device)[valid_tets]
            tet_idx = torch.cat((tet_idx[num_triangles == 1], tet_idx[num_triangles ==
                                                                      2].unsqueeze(-1).expand(-1, 2).reshape(-1)),
                                dim=0)
            return vert_feats, faces, tet_idx

        return vert_feats, faces


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


###############################################################################
#  Geometry interface
###############################################################################
def dot(a, b):
    return torch.einsum("bi,bi->b", a, b)


def get_rotation_matrix(i_v):
    i_v = i_v.clone()
    batch_size = i_v.shape[0]
    unit = torch.tensor([0.,0,1.]).cuda().repeat(batch_size, 1)

    # Normalize vector length
    scale = torch.linalg.norm(i_v, dim=1)
    i_v /= scale.unsqueeze(-1)

    # Get axis
    uvw = torch.cross(i_v, unit,dim=1)

    # compute trig values - no need to go through arccos and back
    rcos = dot(i_v, unit)
    rsin = torch.linalg.norm(uvw, dim=1)

    # Normalize and unpack axis
    mask = ~torch.isclose(rsin, torch.tensor(0.).cuda().repeat(batch_size), atol=1e-6)
    uvw[mask] /= rsin[mask].unsqueeze(-1)

    # Compute rotation matrix - re-expressed to show structure
    u, v, w = uvw.t()

    #tensor(0.) tensor(0.7071) tensor(-0.7071)
    return (
            rcos[:, None, None] * torch.eye(3).cuda().repeat(batch_size, 1, 1) +
            rsin[:, None, None] * torch.stack([
        torch.zeros_like(w), -w, v,
        w, torch.zeros_like(w), -u,
        -v, u, torch.zeros_like(w)
    ]).view(3, 3, batch_size).permute(2, 0, 1) +
            (1.0 - rcos)[:, None, None] * uvw[:, :, None] * uvw[:, None, :]
    ), scale

from tqdm import tqdm
class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS):
        super(DMTetGeometry, self).__init__()

        self.FLAGS = FLAGS
        self.grid_res = grid_res

        cube_range = {32: 3, 64: 4, 128: 5, 256: 6}
        tet = np.load(f'/cluster/work/igp_psr/tpeters/MD_Clusterv2/meshdiffusion/tetra_sorted/from_quartet_4_tets_{cube_range[self.grid_res]}.npz')
        self.verts = torch.tensor(tet['verts'], dtype=torch.float32, device="cuda")
        self.verts += torch.rand_like(self.verts) * 1e-8
        rot_z_90 = RotateAxisAngle(-90, axis="Y", degrees=True).to("cuda")
        self.verts = rot_z_90.transform_points(self.verts)
        # rot_z_90 = RotateAxisAngle(-45, axis="X", degrees=True).to("cuda")
        # self.verts = rot_z_90.transform_points(self.verts)

        if FLAGS.ref_mesh.split("/")[-4] == "03790512":
            print("skaling BIKES!")
            self.verts[:, 1] *= 0.8
            self.verts[:, 0] *= 0.6

        if FLAGS.ref_mesh.split("/")[-4] == "02691156":
            print("skaling Planes!")
            self.verts[:, 1] *= 0.6

        # rot_y_90 = RotateAxisAngle(-95, axis="Y", degrees=True).to("cuda")
        # self.verts = rot_y_90.transform_points(self.verts)
        # rot_x_90 = RotateAxisAngle(-95, axis="X", degrees=True).to("cuda")
        # self.verts = rot_x_90.transform_points(self.verts)
        self.indices = torch.tensor(tet['tet_faces'], dtype=torch.long, device="cuda")
        self.neighbors = torch.tensor(tet['neighbors'], dtype=torch.long, device="cuda")[:, 1:]

        # A15 lattice
        # tets = np.load('data/tets/{}_tets.npz'.format(self.grid_res))
        # self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') #* scale
        # self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')


        self.displacement_prior = torch.zeros(len(self.verts))
        # for i in range(len(self.neighbors)):
        #     node = self.verts[i]
        #     neighbors = self.neighbors[i]
        #     neighbors = self.verts[neighbors[neighbors != -1]]
        #     distances = torch.cdist(node.unsqueeze(0), neighbors)
        #     self.displacement_prior[i] = torch.min(distances)
        # print(self.displacement_prior.max())
        # print(self.displacement_prior.min(), 1/256)
        # exit()
        self.displacement_prior = self.displacement_prior.unsqueeze(-1).cuda()


        self.generate_edges()

        R, s = get_rotation_matrix(self.verts)

        self.rotation_matrices = R
        self.scales = s
        self.marching_tets = DMTet(self.displacement_prior, self.rotation_matrices, self.scales, grid_res=self.grid_res)

        # Random init
        sdf = torch.rand_like(self.verts[:, 0]) - 0.1

        self.sdf = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts[..., :2]), requires_grad=True)
        self.register_parameter('deform', self.deform)

        self.color = torch.nn.Parameter(torch.ones_like(self.verts) * 0.5, requires_grad=True)
        self.register_parameter('color', self.color)

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

    def getMesh(self, material=None):
        # Run DM tet to get a base mesh
        # deform = torch.zeros_like(self.deform)
        # norms = torch.linalg.vector_norm(deform, ord=2, dim=1)
        # mask = norms > self.displacement_prior.squeeze()
        # clipper = torch.ones_like(self.displacement_prior)
        # clipper[mask] = self.displacement_prior[mask, :] / norms[mask].unsqueeze(1)
        # deform = deform * clipper
        # v_deformed = self.verts #+ deform

        # deform = self.verts
        v_deformed = self.verts  # + 2 / (self.grid_res * 2) * torch.tanh(self.deform)
        v_colored = torch.clamp(self.color, 0, 1)
        sdf = self.sdf
        # print(self.color.grad)
        v_feats = torch.cat([v_deformed, v_colored], dim=-1)
        assert v_feats.shape[-1] == (v_colored.shape[-1] + v_deformed.shape[-1])
        feats, faces = self.marching_tets(v_feats, sdf, self.deform, self.indices)
        verts = feats[..., :3]
        colors = feats[..., 3:6]
        # print(feats.shape, faces.shape)
        # print(colors)
        imesh = mesh.Mesh(verts, faces, v_color=colors)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        # imesh = mesh.compute_tangents(imesh)

        return imesh

    def render(self, glctx, target, lgt, opt_material, bsdf=None):
        opt_mesh = self.getMesh(opt_material)
        return opt_mesh, render.custom_render_mesh(glctx, opt_mesh, target['mvp'], target['campos'], lgt,
                                                   target['resolution'], spp=target['spp'],
                                                   msaa=True, background=target['background'], bsdf=bsdf)

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration):
        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        opt_mesh, buffers = self.render(glctx, target, lgt, opt_material)

        # import open3d as o3d
        # vertices = o3d.utility.Vector3dVector(target['orig_mesh'].v_pos.detach().cpu().numpy())
        # faces = o3d.utility.Vector3iVector(target['orig_mesh'].t_pos_idx.detach().cpu().numpy())
        # mesh_o3d = o3d.geometry.TriangleMesh(vertices, faces)
        # # print(target['orig_mesh'].v_pos.mean(dim=0))
        # # print(self.verts.mean(dim=0))
        # pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.verts.detach().cpu().numpy()))
        # o3d.visualization.draw_geometries([mesh_o3d, pcl], point_show_normal=False, mesh_show_back_face=True,
        #                                   mesh_show_wireframe=True)
        # exit()

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:],
                                      color_ref[..., 0:3] * color_ref[..., 3:])
        img_loss = img_loss + 0.1 * torch.nn.functional.mse_loss(buffers['depth'], target['depth'])
        img_loss = img_loss + 0.1 * torch.nn.functional.mse_loss(buffers['mask'], target['mask'])
        # img_loss = img_loss + torch.nn.functional.l1_loss(buffers['normal'], target['normal'])

        # SDF regularizer
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * t_iter)
        reg_loss = sdf_reg_loss(self.sdf, self.all_edges).mean() * sdf_weight  # Dropoff to 0.01

        # Chamferopt_mesh
        # if t_iter > 0.6:
        try:
            meshes = Meshes(verts=list(opt_mesh.v_pos[None, ...]), faces=list(opt_mesh.t_pos_idx[None, ...]))
            pred_points = sample_points_from_meshes(meshes, num_samples=100000, return_normals=False, return_textures=False)
            chamfer = chamfer_distance(pred_points, target['points'])[0]
            self.chamfer = chamfer
            img_loss = img_loss + 1. * chamfer
        except:
            pass
        # if t_iter > 0.6:
        if self.FLAGS.laplace == "absolute":
            reg_loss += 0.1 * regularizer.laplace_regularizer_const(opt_mesh.v_pos, opt_mesh.t_pos_idx) * self.FLAGS.laplace_scale * (t_iter)

        return img_loss, reg_loss
