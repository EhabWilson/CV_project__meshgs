from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union
import pymeshlab as pml
import trimesh
import open3d as o3d
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal

import torch
import torch.distributed
from torch import Tensor
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_apply

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split, _update_param_with_optimizer, meshgs_duplicate, meshgs_split, meshgs_remove
from typing_extensions import Literal
from examples.utils import init_gs_from_mesh, calculate_ellipse_vertices, chamfer_distance, save_mesh, read_points3D_binary, divide_point_cloud, compute_convex_hulls_and_union
from scipy.spatial import KDTree
import numpy as np
from tqdm import tqdm
import os

DEBUG=True

@dataclass
class MeshGSStrategy(Strategy):
    prune_opa: float = 0.05
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    pause_refine_after_reset: int = 0
    refine_scale2d_stop_iter: int = 0
    refine_mesh_stop_iter: int = 15_000
    refine_start_iter: int = 500
    mesh_refine_start_iter: int = 3_000
    refine_stop_iter: int = 15_000
    refine_every: int = 100
    split_every: int = 1000
    reset_every: int = 3000
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    remove_cd: float = 0.01
    remove_cd_norm: float = 0.3
    remove_start_iter: int = 5000
    remove_every: int = 500
    revised_opacity: bool = False
    key_for_gradient: str = "gradient_2dgs"
    verbose: bool = False
    absgrad: bool = False
    
    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None

        return state

    def initialize_mesh_state(self):
        state = {"grad": None,
                 "count":None}
        
        return state
    
    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        info: Dict[str, Any],
    ):
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    @torch.no_grad()
    def _grow_gs(
        self,
        cfg,
        mesh_params: Dict,
        gs_params: Dict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(gs_params["scales"][:,:2]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            meshgs_duplicate(
                mesh_params=mesh_params, 
                gs_params=gs_params, 
                optimizers=optimizers, 
                state=state, 
                mask=is_dupli)

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            meshgs_split(
                cfg=cfg,
                mesh_params=mesh_params, 
                gs_params=gs_params,  
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        mesh_params: Dict,
        gs_params: Dict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:

        is_prune = torch.sigmoid(gs_params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(gs_params["scales"][:,:2]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            meshgs_remove(gs_params=gs_params, mesh_params=mesh_params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune

    def step_post_backward(
        self,
        mesh_params,
        gs_params,
        cfg,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        mesh_state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,    
    ):
        if step >= self.refine_stop_iter:
            return

        n_gaussian = gs_params["means"].shape[0]
        self._update_state(n_gaussian, state, info, packed=packed)

        n_faces = len(mesh_params["faces"])
        self._update_mesh_state(n_faces, gs_params, mesh_state, info)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            if step % self.split_every == 0 and step > self.mesh_refine_start_iter:
                # split triangles
                n_split = self._split_meshes(mesh_params, gs_params, optimizers, state, mesh_state, step, cfg)

                if self.verbose:
                    print(
                        f"Step {step}: {n_split} triangles split. "
                        f"Now having {len(mesh_params['faces'])} triangles."
                    )
                mesh_state['grad'].zero_()
                mesh_state['count'].zero_()
            else:
                # Adaptive Densify Strategy
                n_dupli, n_split = self._grow_gs(cfg, mesh_params, gs_params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                        f"Now having {len(gs_params['means'])} GSs."
                    )
                # prune GSs
                n_prune = self._prune_gs(mesh_params, gs_params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(gs_params['means'])} GSs."
                    )
                    
            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
        
        if (
            step >= self.remove_start_iter
            and step % self.remove_every == 0
        ):
            n_remove = self._remove_transparent_faces(mesh_params, gs_params, state, mesh_state)
            print(
                f"Step {step}: {n_remove} triangles are removed. "
                f"Now having {len(mesh_params['faces'])} triangles."
            )

        if step % self.reset_every == 0:
            reset_opa(
                params=gs_params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )
    
    @torch.no_grad()
    def prune_degradation_faces(
        self,
        cfg,
        gs_params,
        mesh_params: Dict,
        optimizers: Dict[str, torch.optim.Optimizer], 
        state: Dict[str, Any],
        mesh_state: Dict[str, Any],
        ):
        vertices = mesh_params["vertices"]  # [N, 3]
        faces = mesh_params["faces"]        # [F, 3]
        triangles = vertices[faces]         # [F, 3, 3]

        A_in = triangles[:, 0, :] 
        B_in = triangles[:, 1, :] 
        C_in = triangles[:, 2, :]
        a = torch.norm(B_in - C_in, dim=1)
        b = torch.norm(A_in - C_in, dim=1) 
        c = torch.norm(A_in - B_in, dim=1) 

        s = (a + b + c) / 2
        area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

        is_remove = (area <= 1e-8)

        if is_remove.sum() == 0:
            return 
        
        mesh_params["faces"] = faces[~is_remove]
        gs2mesh_index = gs_params["index"]
        sel = ~is_remove[gs2mesh_index]

        # update splats & state        
        for k, v in gs_params.items():
            if isinstance(v, torch.Tensor):
                if k != "index":
                    gs_params[k] = v[sel]
                else:
                    old2new = torch.full((faces.shape[0],), -1, dtype=torch.long, device=faces.device)
                    keep_indices = torch.nonzero(~is_remove).flatten()
                    old2new[keep_indices] = torch.arange(len(keep_indices), device=faces.device)
                    # Update gs_params["index"]
                    gs_params[k] = old2new[v[sel]]  

        app_opt = False if "sh0" in gs_params else True
        if app_opt:
            learanable_names = [
                "uv_sum", "u_ratio",
                "scale_lambda", "rot_2d", 
                "features", "colors", "opacities"
            ]
        else:
            learanable_names = [
                "uv_sum", "u_ratio",
                "scale_lambda", "rot_2d", 
                "sh0", "shN", "opacities"
            ]
        for name in learanable_names:
            gs_params[name] = torch.nn.Parameter(gs_params[name])

        # update the parameters related to gs in the optimizers
        def gs_optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v[sel]
        
        for name in learanable_names:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = gs_optimizer_fn(key, v)
                p_new = gs_params[name]
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
            
        # update states
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v[sel]
        
        # update mesh states
        for k, v in mesh_state.items():
            if isinstance(v, torch.Tensor):
                mesh_state[k] = v[~is_remove]

    @torch.no_grad()
    def _update_mesh_state(
            self,
            n_faces,
            gs_params,
            mesh_state: Dict[str, Any],
            info: Dict[str, Any],
    ):
        faces_grad = info['faces_grad']

        if mesh_state["grad"] is None:
            mesh_state["grad"] = torch.zeros(n_faces, device=gs_params['means'].device)
        if mesh_state["count"] is None:
            mesh_state["count"] = torch.zeros(n_faces, device=gs_params['means'].device)

        if faces_grad is None:
            return

        mesh_state["grad"] += faces_grad
        mesh_state["count"][faces_grad != 0] += 1

    def _update_state(
        self,
        n_gaussian,
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in ["means2d", "width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        if self.key_for_gradient == "gradient_2dgs":
            grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
            grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            if len(grads.shape) != 3:
                grads = grads.unsqueeze(0)
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )
    
    @torch.no_grad()
    def _split_meshes(
        self,
        mesh_params: Dict,
        gs_params: Dict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        mesh_state: Dict[str, Any],
        step: int,
        cfg,
    ) -> int:
        # select faces
        grad = mesh_state['grad']
        count = mesh_state['count'].clamp(min=1)
        device = grad.device

        # is_splited = (grad / count) > cfg.mesh_split_grad_threshold
        if DEBUG:
            save_mesh(mesh_params, f"{cfg.result_dir}/plys/mesh_{step:05d}_before_split.ply")
            np.save(f"{cfg.result_dir}/plys/grad_{step:05d}.npy", (grad / count).detach().cpu().numpy())
        is_splited = torch.zeros(grad.shape[0], device=device)
        is_splited[torch.topk(grad / count, k=int(cfg.mesh_split_topk * grad.shape[0]), dim=0)[1].int()] = 1

        vertices = mesh_params["vertices"]
        faces = mesh_params["faces"]

        if is_splited.sum() == 0:
            return
        
        # get neighbour faces
        splited_face_ids = torch.where(is_splited!=0)[0]
        tm = trimesh.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy())
        face_adjacency = torch.tensor(tm.face_adjacency, dtype=torch.long, device=device)
        adj_faces = face_adjacency[torch.isin(face_adjacency, splited_face_ids).any(axis=1)]
        adj_faces = adj_faces.unique()
        mask = torch.zeros(faces.shape[0]).to(device)
        mask[adj_faces] = 1

        # split mesh
        meshes = pml.MeshSet()
        meshes.add_mesh(pml.Mesh(vertices.cpu().numpy(), faces.cpu().numpy(), f_scalar_array=mask.cpu().numpy()))
        meshes.compute_selection_by_condition_per_face(condselect='fq == 1')
        # meshes.meshing_surface_subdivision_midpoint(iterations=1, threshold=pml.PureValue(state['scene_scale'] * 0.002), selected=True)
        meshes.meshing_surface_subdivision_midpoint(iterations=1, threshold=pml.PureValue(0.01), selected=True)
        
        # update mesh params
        m = meshes.current_mesh()
        new_vertices = torch.tensor(m.vertex_matrix(), dtype=torch.float32).to(device)
        new_faces = torch.tensor(m.face_matrix(), dtype=torch.long, requires_grad=False).to(device)
        mesh_params["faces"] = new_faces
        mesh_params["vertices"] = torch.nn.Parameter(new_vertices).to(device)

        # update the parameters related to mesh in the optimizers
        def mesh_optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.cat([v, torch.zeros((new_vertices.shape[0]-vertices.shape[0], *v.shape[1:]), device=device)])

        for name in ["vertices"]:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = mesh_optimizer_fn(key, v)
                p_new = mesh_params[name]
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state

        # remove gaussians in splited meshes and add gaussians in new added faces
        splited_face_ids = torch.where(mask!=0)[0]
        is_removed = torch.isin(gs_params["index"], splited_face_ids)
        new_face_ids = torch.cat([splited_face_ids, torch.arange(faces.shape[0], new_faces.shape[0]).to(device)])
        new_added_faces = new_faces[new_face_ids]

        feature_dim = 32 if cfg.app_opt else None
        new_gs_params = init_gs_from_mesh(cfg, {"faces": new_added_faces, "vertices": new_vertices}, 
                                            device=device, feature_dim=feature_dim)
        new_added_gs_num = new_face_ids.shape[0] * cfg.num_mesh2gs
        
        # update new_added_gs sh0 & shN or appearance features
        removed_gs_means = gs_params['means'][is_removed]
        removed_gs_opacities = gs_params['opacities'][is_removed]
        new_added_gs_means = new_gs_params['means']

        removed_gs_means_np = removed_gs_means.cpu().numpy()
        new_added_gs_means_np = new_added_gs_means.cpu().numpy()
        kdtree = KDTree(removed_gs_means_np)
        nearest_index_np = kdtree.query(new_added_gs_means_np)[1]
        nearest_index = torch.tensor(nearest_index_np, dtype=torch.long, device=device)

        new_gs_params['opacities'] = removed_gs_opacities[nearest_index]

        if feature_dim is None:
            removed_gs_sh0 = gs_params['sh0'][is_removed]
            removed_gs_shN = gs_params['shN'][is_removed]
            new_gs_params['sh0'] = removed_gs_sh0[nearest_index]
            new_gs_params['shN'] = removed_gs_shN[nearest_index]
        else:
            remove_gs_features = gs_params['features'][is_removed]
            removed_gs_colors = gs_params['colors'][is_removed]
            new_gs_params['features'] = remove_gs_features[nearest_index]
            new_gs_params['colors'] = removed_gs_colors[nearest_index]

        # update gs_params
        if feature_dim is None:
            learanable_names = [
                "uv_sum", "u_ratio",
                "scale_lambda", "rot_2d", 
                "sh0", "shN", "opacities"
            ]
        else:
            learanable_names = [
                "uv_sum", "u_ratio",
                "scale_lambda", "rot_2d", 
                "features", "colors", "opacities"
            ]

        for name in learanable_names:
            gs_params[name] = torch.cat([gs_params[name][~is_removed], new_gs_params[name]])
            gs_params[name] = torch.nn.Parameter(gs_params[name])
        for name in ["means", "scales", "quats"]:
            gs_params[name] = torch.cat([gs_params[name][~is_removed], new_gs_params[name]])
        gs_params["index"] = torch.cat([gs_params["index"][~is_removed], new_face_ids.repeat_interleave(cfg.num_mesh2gs)])

        # update the parameters related to gs in the optimizers
        def gs_optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.cat([v[~is_removed], torch.zeros((new_added_gs_num, *v.shape[1:]), device=device)])
        
        for name in learanable_names:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = gs_optimizer_fn(key, v)
                p_new = gs_params[name]
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state

        # update states
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                v_new = torch.zeros((new_added_gs_num, *v.shape[1:]), device=device)
                state[k] = torch.cat((v[~is_removed], v_new))

        for k,v in mesh_state.items():
            if isinstance(v, torch.Tensor):
                mesh_state[k] = torch.zeros((new_faces.shape[0],*v.shape[1:]), device=device)

        if DEBUG:
            save_mesh(mesh_params, f"{cfg.result_dir}/plys/mesh_{step:05d}_after_split.ply")

        return mask.sum().int()
    
    @torch.no_grad()
    def _remove_transparent_faces(self, mesh_params, gs_params, state, mesh_state):
        device = mesh_params["faces"].device
        
        # get vertices from gaussians and mesh
        means, quats = gs_params["means"], gs_params["quats"]
        scales = torch.exp(gs_params["scales"])[:,:2]
        opacities = torch.sigmoid(gs_params["opacities"])
        sel_gs = opacities > 0.3

        means = means[sel_gs]
        quats = quats[sel_gs]
        scales = scales[sel_gs]

        pts = calculate_ellipse_vertices(means, quats[:,[3,0,1,2]], scales).reshape(1, -1, 3)   # (1, n_pts, 3)
        mesh_vertices = mesh_params["vertices"][mesh_params["faces"]]    # (n_faces, 3, 3)
        mesh_normals = torch.cross(mesh_vertices[:,1] - mesh_vertices[:,0], mesh_vertices[:,2] - mesh_vertices[:,0], dim=1)
        mesh_normals = torch.nn.functional.normalize(mesh_normals, dim=1)   # (n_faces, 3)
        mesh_vertices = mesh_vertices.reshape(1, -1, 3) # (1, n_faces*3, 3)
        # pts = torch.cat([pts, means], dim=0)
        pts_normals = mesh_normals[gs_params["index"]][None]    # (1, n_gs, 3)
        pts_normals = torch.repeat_interleave(pts_normals, 4, dim=1)    # (1, n_gs*4, 3)
        mesh_normals = torch.repeat_interleave(mesh_normals[None], 3, dim=1)    # (1, n_faces*3, 3)

        # decide which faces to remove based on chamfer distance
        cd, cd_norm = chamfer_distance(mesh_vertices, pts, mesh_normals, pts_normals)    # (n_faces*3,)
        is_far_from_gs = ((cd.reshape(-1, 3).mean(dim=1) > self.remove_cd * state['scene_scale'])
                    & (cd_norm.reshape(-1, 3).mean(dim=1) > self.remove_cd_norm))
        has_no_gs = torch.ones(len(mesh_params["faces"]), dtype=torch.bool, device=device)
        has_no_gs[torch.unique(gs_params["index"])] = 0
        transparent_faces = is_far_from_gs & has_no_gs
        if transparent_faces.sum() == 0:
            return 0
        
        # get mapping from old faces to new faces
        n_old_faces = len(mesh_params["faces"])
        n_reserved_faces = int(len(mesh_params["faces"]) - transparent_faces.sum())
        old_face_indices = torch.where(~transparent_faces)[0]
        face_indices_mapping = torch.zeros(n_old_faces, dtype=torch.long, device=device)
        face_indices_mapping[old_face_indices] = torch.arange(n_reserved_faces, device=device)

        # update mesh_params
        mesh_params["faces"] = mesh_params["faces"][~transparent_faces]
        for k,v in mesh_state.items():
            if isinstance(v, torch.Tensor):
                mesh_state[k] = mesh_state[k][~transparent_faces]

        # update gs_params
        gs_params["index"] = face_indices_mapping[gs_params["index"]]

        return transparent_faces.sum()
    
    @torch.no_grad()
    def add_mesh_from_bggs(
        self, 
        cfg,
        step,
        mesh_params, 
        gs_params,
        bggs_params,
        optimizers, 
        state, 
        mesh_state,
        distance_threshold = None,
        radius = None,
        min_neighbors = 10,
        subdivide_iterations = 1
    ):
        assert False, "Should not reach here"
        save_mesh(mesh_params, f"{cfg.result_dir}/plys/mesh_{step:05d}_before_add.ply")
        device = mesh_params['faces'].device

        if distance_threshold is None:
            distance_threshold = state['scene_scale'] / 100.0 * 3
        if radius is None:
            radius = state['scene_scale'] / 100.0 * 3

        # mesh
        vertices = mesh_params["vertices"].detach().cpu().numpy()
        faces = mesh_params["faces"].cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh_pcd = mesh.sample_points_uniformly(number_of_points=1000000)
        
        # SfM points
        bg_means, bg_quats, bg_scales, bg_opacities = bggs_params["bg_means"], bggs_params["bg_quats"], bggs_params["bg_scales"], bggs_params["bg_opacities"]
        is_not_transparent = torch.sigmoid(bg_opacities) > 0.9
        bg_means = bg_means[is_not_transparent]
        bg_quats = bg_quats[is_not_transparent]
        bg_scales = torch.exp(bg_scales[is_not_transparent])

        # sample more points from big gs
        extra_points = []
        unit_scale = state['scene_scale'] / 150
        for i, scale in enumerate(bg_scales):
            pos2d = []
            if scale[0] > unit_scale:
                sample_x = torch.linspace(0, scale[0], int(scale[0]//unit_scale)+1)[1:] # (n,)
                pos2d_x = torch.cat([sample_x, -sample_x])  # (2n,)
                pos2d.append(torch.stack([pos2d_x, torch.zeros_like(pos2d_x)]).t()) # (2n, 2)
            if scale[1] > unit_scale:
                sample_y = torch.linspace(0, scale[1], int(scale[1]//unit_scale)+1)[1:] # (n,)
                pos2d_y = torch.cat([sample_y, -sample_y])  # (2n,)
                pos2d.append(torch.stack([torch.zeros_like(pos2d_y), pos2d_y]).t()) # (2n, 2)
            if len(pos2d) != 0:
                pos2d = torch.cat(pos2d)    # (m, 2)
                pos3d = F.pad(pos2d, (0, 1)).to(device)    # (m, 3)
                extra_points.append(bg_means[i:i+1] + quaternion_apply(bg_quats[i:i+1], pos3d)) # (m, 3)
        if len(extra_points) > 0:
            extra_points = torch.cat(extra_points)
            bggs_pcd = o3d.geometry.PointCloud()
            bggs_pcd.points = o3d.utility.Vector3dVector(bg_means.cpu().numpy())
            o3d.io.write_point_cloud(f"{cfg.result_dir}/plys/ply_{step:05d}_origin.ply", bggs_pcd)
            bg_means = torch.cat([bg_means, extra_points])
        
        bg_means = bg_means.cpu().numpy()
        bggs_pcd = o3d.geometry.PointCloud()
        bggs_pcd.points = o3d.utility.Vector3dVector(bg_means)
        o3d.io.write_point_cloud(f"{cfg.result_dir}/plys/ply_{step:05d}_added.ply", bggs_pcd)

        # stage 1: filter points far from mesh
        mesh_kdtree = o3d.geometry.KDTreeFlann(mesh_pcd)
        filtered_points = []
        for point in bggs_pcd.points:
            [k, idx, dist] = mesh_kdtree.search_knn_vector_3d(point, 1)
            if dist[0] > distance_threshold ** 2:
                filtered_points.append(point)

        # stage 2: remove outliers of the filtered points
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
        clean_pcd, _ = filtered_pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)

        # stage 3: convert points to mesh by union of convex hulls
        blocks = divide_point_cloud(np.asarray(clean_pcd.points), vertices.min(axis=0), vertices.max(axis=0), 16)
        combined_hulls = compute_convex_hulls_and_union(blocks)
        
        # subdivide meshes
        meshes = pml.MeshSet()
        meshes.add_mesh(pml.Mesh(np.asarray(combined_hulls.vertices), np.asarray(combined_hulls.faces)))
        meshes.meshing_remove_null_faces()
        meshes.meshing_decimation_clustering()
        meshes.meshing_repair_non_manifold_edges()
        meshes.meshing_surface_subdivision_midpoint(iterations=subdivide_iterations)
        meshes.save_current_mesh(f"{cfg.result_dir}/plys/mesh_{step:05d}_added.ply")

        # update mesh params
        m = meshes.current_mesh()
        new_added_vertices = np.asarray(m.vertex_matrix(), dtype=np.float32)
        new_added_faces = np.asarray(m.face_matrix(), dtype=np.int32)
        new_added_triangles = new_added_vertices[new_added_faces]
        face_area = np.linalg.norm(
            np.cross(new_added_triangles[:,1] - new_added_triangles[:,0], 
                     new_added_triangles[:,2] - new_added_triangles[:,0]),
            axis=1
        )
        new_added_faces = new_added_faces[face_area > 1e-8]
        new_added_faces += len(mesh_params["vertices"])

        new_vertices = torch.concat((mesh_params["vertices"], torch.tensor(new_added_vertices, device=device)), dim=0)
        new_faces = torch.concat((mesh_params["faces"], torch.tensor(new_added_faces, device=device)), dim=0)
        mesh_params["vertices"] = torch.nn.Parameter(new_vertices).to(device)
        mesh_params['faces'] = new_faces.to(device)

        # update the parameters related to mesh in the optimizers
        def mesh_optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.cat([v, torch.zeros((new_added_vertices.shape[0], *v.shape[1:]), device=device)])

        for name in ["vertices"]:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = mesh_optimizer_fn(key, v)
                p_new = mesh_params[name]
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state     

        new_gs_params = init_gs_from_mesh(cfg, {"faces": new_added_faces, "vertices": new_vertices}, device)

        # init sh0
        kdtree = KDTree(gs_params['means'].cpu().numpy())
        nearest_index_np = kdtree.query(new_gs_params['means'].cpu().numpy())[1]
        nearest_index = torch.tensor(nearest_index_np, dtype=torch.long, device=device)
        new_gs_params['sh0'] = gs_params['sh0'][nearest_index]

        # update gs params
        for name in ["uv_sum", "u_ratio", "rot_2d", "sh0", "shN", "opacities", "scale_lambda"]:
            gs_params[name] = torch.cat([gs_params[name], new_gs_params[name]])
            gs_params[name] = torch.nn.Parameter(gs_params[name])
        for name in ["means", "scales", "quats"]:
            gs_params[name] = torch.cat([gs_params[name], new_gs_params[name]])
        gs_params["index"] = torch.cat([gs_params["index"], torch.arange(len(new_faces) - len(new_added_faces), len(new_faces), device=device)])

        new_added_gs_num = new_added_faces.shape[0]
        # update the parameters related to gs in the optimizers
        def gs_optimizer_fn(key: str, v: Tensor) -> Tensor:
            return torch.cat([v, torch.zeros((new_added_gs_num, *v.shape[1:]), device=device)])
        
        for name in ["uv_sum", "u_ratio", "rot_2d", "sh0", "shN", "opacities", "scale_lambda"]:
            optimizer = optimizers[name]
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        p_state[key] = gs_optimizer_fn(key, v)
                p_new = gs_params[name]
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
        
        # update states
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                v_new = torch.zeros((new_added_gs_num, *v.shape[1:]), device=device)
                state[k] = torch.cat((v, v_new))

        for k,v in mesh_state.items():
            if isinstance(v, torch.Tensor):
                v_new = torch.zeros((new_added_faces.shape[0],*v.shape[1:]), device=device)
                mesh_state[k] = torch.cat((v, v_new))