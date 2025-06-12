import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import imageio
import open3d as o3d
import gc
import nerfview
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (AppearanceOptModule, CameraOptModule, knn, read_points3D_binary,
                   rgb_to_sh, set_random_seed, colormap, apply_depth_colormap, 
                   l1_loss, focal_l1_loss, tv_loss, sky_loss, save_ply, merge_ply,
                   save_image, depth_gray2rgb, normal_rgb2sn, post_process_mesh, ssim, 
                   least_squares, dice_loss, consistency_loss, depth_ranking_loss, chamfer_distance,
                   init_colors_from_pc, triangles_to_ellipses, compute_quats,
                   edge_equilateral_triangle_loss, rotation_matrix_to_quaternion, calculate_means, calculate_scales, calculate_Rs,
                   read_points3D_binary,)
from collections import defaultdict

from mesh_opt import mesh_merge
from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import MeshGSStrategy, DefaultStrategy
from gsplat.utils import normalized_quat_to_rotmat
from matplotlib import pyplot as plt

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_edge_loss, mesh_normal_consistency, mesh_laplacian_smoothing

import cv2

torch.autograd.set_detect_anomaly(True)

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[List[str]] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # Load sky mask
    sky_mask_on: bool = False
    # Weight for sky loss
    sky_lambda: float = 1e-3
    # Load dynamic mask
    dynamic_mask_on: bool = False

    # Load instance mask
    instance_map_on: bool = False

    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000

    # Number of position lr steps
    pos_lr_steps: int = 30_000

    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05

    grow_scale3d: float = 0.01

    grow_grad2d: float = 0.0002
    
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Pause refining GSs until this number of steps after reset
    pause_refine_after_reset: int = 0

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for consistency normal loss
    normal_consistency_lambda: float = 5e-2
    # Weight for smooth normal loss
    normal_smooth_lambda: float = 2e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 7_000

    # Load mono depth
    mono_depth_loss: bool = False
    # Iteration to start mono_depth_regularization
    mono_depth_start_iter: int = 5_000
    # Weight for mono depth loss
    mono_depth_lambda: float = 0.5

    # Load plane masks
    plane_consistency_loss: bool = False
    # Weight for plane normal consistency
    plane_consistency_lambda: float = 1e-1
    # Load normal map
    normal_map_on: bool = False
    # use normal map loss
    plane_normal_map_loss: bool = False
    # Weight for normal map loss
    plane_normal_map_lambda: float = 1e-1
    # Iteration to start plane normal map regularization
    plane_normal_map_loss_start_iter: int = 5_000

    # Edge Loss
    edge_loss: bool = False
    # Weight for edge loss
    edge_lambda: float = 1e-2

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000
    
    # "focal" l1 loss, this leads to better details (e.g. fence) but can be sensitive to noise
    focal_loss_on: bool = False

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Log with wandb
    wandb_on: bool = False
    # Dump information to tensorboard every this steps
    log_every: int = 50
    # Save training images to tensorboard
    log_save_image: bool = False

    # Depth rank mode
    depth_rank_mode: Literal["random_sample", "global"] = "global"
    # Depth rank loss
    depth_rank_loss: bool = False
    # Weight for depth rank loss
    depth_rank_lambda: float = 1

    # Num of sampled point pairs
    num_sampled_pairs: int = 4096

    # Dust3r
    dust3r_init: bool = False

    # Opacity
    opacity_ce_loss: bool = False
    # Weight for opacity cross entropy loss
    opacity_ce_lambda: float = 1e-2

    # regularization for triangles
    mesh_triangle_reg: bool = False
    mesh_triangle_reg_lambda: float = 1e-2

    # rhombus_matching_loss
    rhombus_matching_loss: bool = False
    rhombus_matching_lambda: float = 1e-2

    # mesh loss
    mesh_loss: bool = False
    mesh_loss_start_iter: int = 3000
    mesh_edge_consistency_lambda: float = 1
    mesh_normal_consistency_lambda: float = 0.01
    mesh_smooth_lambda: float = 0.1

    # mesh prune
    mesh_prune_opacity_threshold: float = 0.005

    freeze_steps: int = 3000

    path_to_mesh: str = ""

    # init num2gs
    num_mesh2gs: int = 1

    # learning rate for parameters
    lr_vertices: float = 1.6e-4

    lr_uvsum: float = 0.01

    lr_uratio: float = 0.01

    lr_scalelambda: float = 0.01

    lr_quat: float = 1e-3

    lr_rot2d: float = 1e-2
    
    lr_sh0: float = 2.5e-3
    
    lr_shN: float = 2.5e-3 / 20
    
    lr_opa: float = 5e-2

    # BG GS settingsss
    bg_gs_start_iter: int = 3000
    bg_gs_stop_iter: int = 7000
    
    # remove mesh faces settings
    remove_cd: float = 0.001
    remove_cd_norm: float = 0.00
    remove_every: int = 500
    remove_start_iter: int = 5000

    min_rel_scale: float = 0.5
    max_rel_scale: float = 1.5

    # mesh split
    mesh_refine_start_iter: int = 3_000

    mesh_split_topk: float = 0.03

    split_every: int = 1000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.pos_lr_steps = int(self.pos_lr_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
        self.normal_start_iter = int(self.normal_start_iter * factor)
        self.mono_depth_start_iter = int(self.mono_depth_start_iter * factor)

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    cfg = None,
    mesh_gs = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    sfm_bin_path = os.path.join(cfg.data_dir,'sparse/0/points3D.bin')
    points, rgbs, _ = read_points3D_binary(sfm_bin_path, max_err=2, min_track_length=3)
    points = torch.from_numpy(points).float().to(device)
    rgbs = torch.from_numpy(rgbs / 255.0).float().to(device)
    normals = None
    
    if mesh_gs is not None:
        cd = chamfer_distance(points[None], mesh_gs[None])[0][0]
        mask = cd > 0.1
        points = points[mask]
        rgbs = rgbs[mask]

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    if normals is None:
        quats = torch.rand((N, 4))  # [N, 4]
    else:
        # quats = torch.rand((N, 4))
        quats = compute_quats(normals[world_rank::world_size], points)

    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("bg_means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("bg_scales", torch.nn.Parameter(scales), 5e-3),
        ("bg_quats", torch.nn.Parameter(quats), 1e-3),
        ("bg_opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("bg_sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("bg_shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("bg_features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("bg_colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(BS)}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def create_mesh_anchors_with_optimizers(
    parser: Parser,
    cfg: Config,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # init mesh peoperties
    mesh = o3d.io.read_triangle_mesh(cfg.path_to_mesh)
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()

    vertices = torch.nn.Parameter(
        torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device, requires_grad=True)
    )  # [N, 3]

    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long, device=device, requires_grad=False)  # [F, 3]
    
    params = [
        ("vertices", torch.nn.Parameter(vertices), cfg.lr_vertices * scene_scale),
    ]

    mesh_anchors = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.Adam)(
            [{"params": mesh_anchors[name], "lr": lr * math.sqrt(BS)}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return mesh_anchors, faces, optimizers

def init_gs_from_mesh(
        parser: Parser,
        cfg: Config, 
        mesh_params,
        feature_dim: Optional[int] = None,
        device="cuda"):
    """
    mesh2gs: convert mehs faces to gs ellipses
    
    @Learnable parameters:
    1. lambda: [N,2] -> means = A + u·AB + v·AC, where u + v <= 1
    2. scale_lambda: [N, 2]
    3. rot_2d: [N, 2]
    4. opacities: [N]
    5. sh0: [N, 1]
    6. shN: [N, 15]
    
    @Intermediate Parameters:
    1. means: [N, 3]
    2. scales: [N, 3]
    3. quats: [N, 4]

    @ Index
    1. index: [N]
    """
    # points cloud
    init_points = torch.from_numpy(parser.points).float().to(device)             # [n, 3]
    init_colors = torch.from_numpy(parser.points_rgb / 255.0).float().to(device) # [n, 3]

    num_faces = mesh_params["faces"].shape[0]

    faces = mesh_params["faces"]
    vertices = mesh_params["vertices"]
    triangles = vertices[faces]

    assert cfg.num_mesh2gs == 1
    means = calculate_means(triangles)
    scales = calculate_scales(triangles)
    Rs = calculate_Rs(triangles)

    ####################### learnable parameters #######################
    A = vertices[faces[:, 0], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    B = vertices[faces[:, 1], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    C = vertices[faces[:, 2], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    AB = B - A
    AC = C - A

    AO = means - A

    M = torch.stack([AB, AC], dim=-1)  # [F, 3, 2]

    uv = torch.linalg.lstsq(M, AO.unsqueeze(-1)).solution.squeeze(-1)

    u = uv[:, :1]    # [F, 1] 
    v = uv[:, 1:]    # [F, 1]

    uv_sum = torch.logit((u + v).clamp(0.0001, 0.9999))
    u_ratio = torch.logit((u / (u + v)).clamp(0.0001, 0.9999))

    scale_lambda =  torch.logit(0.5 * torch.ones((num_faces * cfg.num_mesh2gs, 2), dtype=torch.float, device=device))

    rot_2d = torch.Tensor([[1, 0]]).repeat(num_faces * cfg.num_mesh2gs, 1).to(device)

    opacities = torch.logit(cfg.init_opa * torch.ones((num_faces * cfg.num_mesh2gs), dtype=torch.float, device=device))   # [F]

    params = {
        "uv_sum": uv_sum,
        "u_ratio": u_ratio,
        "scale_lambda": scale_lambda,
        "rot_2d": rot_2d,
        "opacities": opacities
    }

    if feature_dim is None:
        N = means.shape[0]
        rgbs = init_colors_from_pc(init_points, init_colors, means)
        colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)

        sh0 = colors[:, :1, :].to(device)
        shN = colors[:, 1:, :].to(device)

        params.update({
            "sh0": sh0,
            "shN": shN,
        })
    else:
        N = means.shape[0]
        rgbs = init_colors_from_pc(init_points, init_colors, means)
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        colors = torch.logit(rgbs)  # [N, 3]
        params.update({
            "features":  features,
            "colors":  colors,
        })

    params = {n: torch.nn.Parameter(v.to(device)) for n, v in params.items()}

    ####################### Intermediate parameters #######################
    scale_range = cfg.max_rel_scale - cfg.min_rel_scale
    scales = torch.cat([scales * (torch.sigmoid(scale_lambda) * scale_range + cfg.min_rel_scale), torch.ones_like(scales)[:,:1]], dim=-1)
    scales = torch.log(scales)

    normalized_rot_2d = F.normalize(rot_2d, dim=-1)
    R_0 = normalized_rot_2d[..., 0:1] * Rs[..., 0] + normalized_rot_2d[..., 1:2] * Rs[..., 1]
    R_1 = -normalized_rot_2d[..., 1:2] * Rs[..., 0] + normalized_rot_2d[..., 0:1] * Rs[..., 1]
    R_2 = Rs[..., 2]
    R = torch.cat([R_0[..., None], R_1[..., None], R_2[..., None]], dim=-1)
    quats = rotation_matrix_to_quaternion(R)

    gs2mesh_index = torch.arange(num_faces, dtype=torch.long, device=device).unsqueeze(-1).repeat(1, cfg.num_mesh2gs).reshape(-1)

    params.update({
        "means": means,
        "scales": scales,
        "quats": quats,
        "index": gs2mesh_index
    })
    
    return params


def update_gs(cfg: Config, mesh_params, gs_params, optimizers, world_size):

    faces = mesh_params["faces"]
    vertices = mesh_params["vertices"]
    gs2mesh_index = gs_params["index"]
    
    # update gs means & quats based on leanable paramss
    uv_sum = gs_params["uv_sum"]
    u_ratio = gs_params["u_ratio"]
    u = torch.sigmoid(uv_sum) * torch.sigmoid(u_ratio)
    v = torch.sigmoid(uv_sum) - u

    triangles = vertices[faces][gs2mesh_index]

    means = (
        triangles[:,0]
        + u * (triangles[:,1] - triangles[:,0])
        + v * (triangles[:,2] - triangles[:,0])
    )

    scale_lambda = gs_params["scale_lambda"]
    scales = calculate_scales(triangles)
    
    scale_range = cfg.max_rel_scale - cfg.min_rel_scale
    scales = torch.cat([scales * (torch.sigmoid(scale_lambda) * scale_range + cfg.min_rel_scale), torch.ones_like(scales)[:,:1]], dim=-1)
    scales = torch.log(scales)

    Rs = calculate_Rs(triangles)
    rot_2d = gs_params["rot_2d"]
    normalized_rot_2d = F.normalize(rot_2d, dim=-1)
    R_0 = normalized_rot_2d[..., 0:1] * Rs[..., 0] + normalized_rot_2d[..., 1:2] * Rs[..., 1]
    R_1 = -normalized_rot_2d[..., 1:2] * Rs[..., 0] + normalized_rot_2d[..., 0:1] * Rs[..., 1]
    R_2 = Rs[..., 2]
    R = torch.cat([R_0[..., None], R_1[..., None], R_2[..., None]], dim=-1)
    quats = rotation_matrix_to_quaternion(R)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats
    }

    gs_params.update(params)

class Runner:
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config, run) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.run = run
        self.distributed = world_size > 1

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/plys"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.eval_dir = f"{cfg.result_dir}/eval"
        os.makedirs(self.eval_dir, exist_ok=True)
        self.vis_dir = f"{cfg.result_dir}/vis"
        os.makedirs(self.vis_dir, exist_ok=True)

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=False,
            dust3r_init=cfg.dust3r_init,
            test_every=cfg.test_every,
            sky_mask_on=cfg.sky_mask_on,
            dynamic_mask_on=cfg.dynamic_mask_on,
            mono_depth_on=cfg.mono_depth_loss or cfg.depth_rank_loss,
            load_plane_masks=cfg.plane_consistency_loss,
            load_normal_maps=cfg.normal_map_on,
            instance_map_on= cfg.instance_map_on
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        mesh_anchors, faces, self.optimizers = create_mesh_anchors_with_optimizers(
            self.parser,
            cfg,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )

        self.mesh_params = {**mesh_anchors, "faces": faces}
        
        self.splats = init_gs_from_mesh(self.parser, 
                                        self.cfg, 
                                        mesh_params=self.mesh_params, 
                                        feature_dim=feature_dim,
                                        device=self.device)

        self.bg_splats = None
        
        splats_params = [
            ("uv_sum", self.splats["uv_sum"], cfg.lr_uvsum),
            ("u_ratio", self.splats["u_ratio"], cfg.lr_uratio),
            ("scale_lambda", self.splats["scale_lambda"], cfg.lr_scalelambda),
            ("rot_2d", self.splats["rot_2d"], cfg.lr_rot2d),
            ("opacities", self.splats["opacities"], cfg.lr_opa),
        ]
        if cfg.app_opt:
            splats_params += [
                ("features",  self.splats["features"], 2.5e-3),
                ("colors", self.splats["colors"], 2.5e-3),
                ]
        else:
            splats_params += [
                ("sh0", self.splats["sh0"], cfg.lr_sh0),
                ("shN", self.splats["shN"], cfg.lr_shN),
            ]

        BS = cfg.batch_size * world_size
        self.optimizers.update({
            name: (torch.optim.Adam)(
                [{"params": params, "lr": lr * math.sqrt(BS)}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            ) for name, params, lr in splats_params
        })
        
        # DEBUG_TMP
        print("Saving initial Gaussians")
        save_ply(os.path.join(self.cfg.result_dir,"plys/init.ply"), self.splats)
        
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        assert cfg.reset_every >= cfg.pause_refine_after_reset, "[ERROR] reset_every must be >= pause_refine_after_reset."
        # Strategy
        self.strategy = MeshGSStrategy(
            verbose=True,
            grow_scale3d=cfg.grow_scale3d,
            grow_grad2d=cfg.grow_grad2d,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            refine_every=cfg.refine_every,
            reset_every=cfg.reset_every,
            split_every=cfg.split_every,
            mesh_refine_start_iter=cfg.mesh_refine_start_iter,
            remove_cd=cfg.remove_cd,
            remove_cd_norm=cfg.remove_cd_norm,
            remove_every=cfg.remove_every,
            remove_start_iter=cfg.remove_start_iter,
            key_for_gradient=key_for_gradient,
        )

        # self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(
            scene_scale = self.scene_scale
        )
        self.mesh_state = self.strategy.initialize_mesh_state()

        # Metrics & Loss
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Losses & Metrics.

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        if self.bg_splats is not None:
            means = torch.cat([self.splats["means"], self.bg_splats["bg_means"]])  # [N, 3]
            # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
            # rasterization does normalization internally
            quats = torch.cat([self.splats["quats"], self.bg_splats["bg_quats"]])  # [N, 4]
            scales = torch.exp(torch.cat([self.splats["scales"], self.bg_splats["bg_scales"]]))  # [N, 3]
            opacities = torch.sigmoid(torch.cat([self.splats["opacities"], self.bg_splats["bg_opacities"]]))  # [N,]
        else:
            means = self.splats["means"]  # [N, 3]
            quats = self.splats["quats"]  # [N, 4]
            scales = torch.exp(self.splats["scales"])  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        
        image_ids = kwargs.pop("image_ids", None)
        val_image_ids = kwargs.pop("val_image_ids", None)

        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                val_image_ids=val_image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

            if self.bg_splats is not None:
                bg_colors = torch.cat([self.bg_splats["bg_sh0"], self.bg_splats["bg_shN"]], 1)  # [N, K, 3]
                colors = torch.cat([colors, bg_colors])

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        if self.model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        elif self.model_type == "2dgs-inria":
            rasterize_fnc = rasterization_2dgs_inria_wrapper

        renders, info = rasterize_fnc(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.distributed,
            **kwargs,
        )

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                render_depths,
            ) = renders
        elif self.model_type == "2dgs-inria":
            render_colors, render_depths, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            render_depths,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.json", "w") as f:
                json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            # TODO vertices schedule
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["vertices"], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
            ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
            # ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if step % 100 == 0:
                torch.cuda.empty_cache()
            
            if step == cfg.bg_gs_start_iter:
                # loading bg gs
                feature_dim = 32 if cfg.app_opt else None
                bg_splats, bg_optimizers = create_splats_with_optimizers(
                    self.parser,
                    init_type=cfg.init_type,
                    init_num_pts=cfg.init_num_pts,
                    init_extent=cfg.init_extent,
                    init_opacity=cfg.init_opa,
                    init_scale=cfg.init_scale,
                    scene_scale=self.scene_scale,
                    sh_degree=cfg.sh_degree,
                    sparse_grad=cfg.sparse_grad,
                    batch_size=cfg.batch_size,
                    feature_dim=feature_dim,
                    device=self.device,
                    world_rank=world_rank,
                    world_size=world_size,
                    cfg=cfg,
                    mesh_gs=self.splats["means"]
                )
                self.bg_splats = bg_splats
                self.optimizers.update(bg_optimizers)
                save_ply(os.path.join(self.cfg.result_dir,"plys/init_bg.ply"), self.bg_splats, is_bg=True)
                print("Model initialized. Number of background GS:", len(self.bg_splats["bg_means"]))

                self.bg_strategy = DefaultStrategy(
                    verbose=True,
                    prune_opa=cfg.prune_opa,
                    grow_grad2d=cfg.grow_grad2d,
                    grow_scale3d=cfg.grow_scale3d,
                    # prune_scale3d=cfg.prune_scale3d,
                    pause_refine_after_reset=cfg.pause_refine_after_reset,
                    refine_start_iter=cfg.refine_start_iter,
                    refine_stop_iter=cfg.refine_stop_iter,
                    reset_every=cfg.reset_every,
                    refine_every=cfg.refine_every,
                    absgrad=cfg.absgrad,
                    revised_opacity=cfg.revised_opacity,
                    key_for_gradient="gradient_2dgs",
                )

                self.bg_strategy_state = self.bg_strategy.initialize_state(
                    scene_scale = self.scene_scale
                )
            
            if step == cfg.bg_gs_stop_iter:
                self.strategy.add_mesh_from_bggs(
                    cfg,
                    step,
                    self.mesh_params,
                    self.splats,
                    self.bg_splats,
                    self.optimizers,
                    self.strategy_state,
                    self.mesh_state,
                )
                self.bg_splats = None

            if step < cfg.freeze_steps:
                self.mesh_params["vertices"].requires_grad = False
            else:
                self.mesh_params["vertices"].requires_grad = True

            update_gs(cfg, self.mesh_params, self.splats, self.optimizers, world_size)

            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # get valid mask
            h, w = pixels.shape[1:3]
            mask = torch.ones((1, h, w, 1), dtype=torch.bool).to(device)

            if self.cfg.sky_mask_on:
                smask = data["smask"].to(device)
                mask = torch.logical_and(mask, torch.logical_not(smask))

            if self.cfg.dynamic_mask_on:
                dmask = data["dmask"].to(device)
                mask = torch.logical_and(mask, dmask)
            
            # get mono depth
            if self.cfg.mono_depth_loss or self.cfg.depth_rank_loss:
                mono_depths = data["mono_depths"].to(device)

            # get plane masks
            if self.cfg.plane_consistency_loss:
                plane_mask = data["plane_mask"].to(device)

            if self.cfg.plane_normal_map_loss or self.cfg.normal_map_on:
                normal_map = data["normal_map"].to(device)
            
            # forward
            (
                renders,
                alphas,
                render_normals,
                render_normals_from_depth,
                render_distort,
                render_median,
                render_depths,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
            )
            
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                info=info,
            )
            
            # loss
            if cfg.focal_loss_on:
                l1loss = focal_l1_loss(colors, pixels, mask)
            else:
                l1loss = l1_loss(colors, pixels, mask)
            
            ssimloss = 1.0 - ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), mask=mask)

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            
            if cfg.mesh_loss and step >= cfg.mesh_loss_start_iter:
                mesh = Meshes(verts=[self.mesh_params["vertices"]], faces=[self.mesh_params["faces"]])
                mesh_edge_consistency_loss = mesh_edge_loss(mesh)
                loss += cfg.mesh_edge_consistency_lambda * mesh_edge_consistency_loss
                
                mesh_normal_consistency_loss = mesh_normal_consistency(mesh)
                loss += cfg.mesh_normal_consistency_lambda  * mesh_normal_consistency_loss

                mesh_smooth_loss = mesh_laplacian_smoothing(mesh)
                loss += (cfg.mesh_smooth_lambda / self.strategy_state['scene_scale']) * mesh_smooth_loss

            if cfg.mono_depth_loss and step > cfg.mono_depth_start_iter:
                curr_mono_depth_lambda = cfg.mono_depth_lambda
                b, h, w, _, _ = mono_depths.shape
                # mono_depths : [B,H,W,1]
                # render_depths : [B,H,W,1]

                # mask out dynamic and sky
                mono_depths = mono_depths.reshape(-1, h * w)[mask.reshape(-1 , h * w)].reshape(b, -1)
                render_depths = render_depths.squeeze(-1).reshape(-1, h * w)[mask.reshape(-1, h * w)].reshape(b,-1)
                
                # remove far depths
                mono_depths_threshold = torch.quantile(mono_depths, 0.1, dim=-1, keepdim=True)
                mono_depths_mask = mono_depths >= mono_depths_threshold
                render_depths_threshold = torch.quantile(render_depths, 0.9, dim=-1, keepdim=True)
                render_depths_mask = render_depths <= render_depths_threshold

                final_mask = torch.logical_and(mono_depths_mask, render_depths_mask)
                mono_depths = mono_depths[final_mask].reshape(b, -1)
                render_depths = render_depths[final_mask].reshape(b, -1)

                least_squares_results = least_squares(mono_depths, render_depths)
                scale = least_squares_results[:, 0].unsqueeze(-1)   # [B, 1]
                bias = least_squares_results[:, 1].unsqueeze(-1)    # [B, 1]

                mono_depths = scale * mono_depths + bias
                pdepth_loss = torch.abs(mono_depths - render_depths).mean() * curr_mono_depth_lambda

                loss = loss + pdepth_loss
            
            if cfg.depth_rank_loss:
                if step > cfg.mono_depth_start_iter:
                    curr_depth_rank_lambda = cfg.depth_rank_lambda
                else:
                    curr_depth_rank_lambda = 0.0
                margin = 1e-4
                if cfg.depth_rank_mode == "random_sample":
                    # predefined parameters
                    box_h = 50
                    box_w = 60
                    # sample several pairs of points
                    h_r = np.random.randint(h-box_h+1, size=(self.cfg.num_sampled_pairs, 1))
                    w_r = np.random.randint(w-box_w+1, size=(self.cfg.num_sampled_pairs, 1))
                    h_r_delta = np.random.randint(box_h, size=(self.cfg.num_sampled_pairs, 2))
                    w_r_delta = np.random.randint(box_w, size=(self.cfg.num_sampled_pairs, 2))
                    h_r = np.concatenate((h_r+h_r_delta[:,:1],h_r+h_r_delta[:,1:2]), axis=1)
                    w_r = np.concatenate((w_r+w_r_delta[:,:1],w_r+w_r_delta[:,1:2]), axis=1)
                    mono_depths_0 = mono_depths.reshape(-1, h*w)[:, h_r[:,0]*w+w_r[:,0]]
                    mono_depths_1 = mono_depths.reshape(-1, h*w)[:, h_r[:,1]*w+w_r[:,1]]
                    render_depths_0 = render_depths.reshape(-1, h*w)[:, h_r[:,0]*w+w_r[:,0]]
                    render_depths_1 = render_depths.reshape(-1, h*w)[:, h_r[:,1]*w+w_r[:,1]]
                    # get depth rank loss (Note: value of mono_depths is larger when closer)
                    depths_rank = torch.where(mono_depths_0>mono_depths_1, 1, -1)   # 1 means mono_depths_0 is closer
                    depth_rank_loss = torch.mean(torch.maximum(
                        depths_rank * (render_depths_0 - render_depths_1) - margin, torch.tensor(0)
                    ))
                elif cfg.depth_rank_mode == "global":
                    depth_rank_loss = depth_ranking_loss(render_depths, mono_depths, 3, 1)
                loss += depth_rank_loss * curr_depth_rank_lambda

            # plane normals with pseudo normals
            plane_normal_map_loss = torch.tensor(0.0, device=device)
            if self.cfg.plane_normal_map_loss and step >= self.cfg.plane_normal_map_loss_start_iter:
                normal_map = normal_map @ camtoworlds[0, :3, :3].T      # Transform to world coordinate
                normal_map = normal_map.squeeze(0)                      # [H, W, 3]

                normals = render_normals.clone().squeeze(0)             # [H, W, 3]: world coordinate
                normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)  # normalize

                abs_diff = torch.abs(normal_map * mask - normals * mask)
                alpha = 2.0
                beta = 2.0
                focal_weight = alpha * (1 - torch.exp(-beta * abs_diff))
                weighted_diff = focal_weight * abs_diff
                plane_normal_map_loss = self.cfg.plane_normal_map_lambda * torch.mean(weighted_diff)
                
                loss += plane_normal_map_loss
            
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_consistency_lambda = cfg.normal_consistency_lambda
                    curr_normal_smooth_lambda = cfg.normal_smooth_lambda
                else:
                    curr_normal_consistency_lambda = curr_normal_smooth_lambda = 0.0
                normals = render_normals.clone()
                normals_from_depth = render_normals_from_depth.clone()

                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                # normal consistency loss
                normal_error_consistency = (1 - (normals * normals_from_depth).sum(dim=0) * mask.squeeze())
                normalloss_consistency = normal_error_consistency.mean()
                
                # normal smooth loss
                normalloss_smooth = tv_loss(normals, mask.squeeze(3))
                
                normalloss = (normalloss_consistency * curr_normal_consistency_lambda
                              + normalloss_smooth * curr_normal_smooth_lambda)
                loss += normalloss

            if cfg.sky_mask_on:
                skyloss = sky_loss(alphas.clamp(1e-6, 1.0).squeeze(), smask.squeeze())
                loss += skyloss * cfg.sky_lambda
            
            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda
            
            loss.backward()
            
            # accumulate faces grads
            if self.mesh_params['vertices'].grad is not None:
                vertices = self.mesh_params['vertices'].clone().detach()
                faces = self.mesh_params['faces'].clone().detach()
                vertices_grad = self.mesh_params['vertices'].grad.clone().detach()
                faces_vertices_grad = vertices_grad[faces]
                faces_vertices = vertices[faces]
                
                # projection
                edge1 = faces_vertices[:, 1] - faces_vertices[:, 0]
                edge2 = faces_vertices[:, 2] - faces_vertices[:, 0]
                normals = torch.cross(edge1, edge2, dim=1)
                normals_norm = normals.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
                normals = normals / normals_norm            # [F,3]
                normals_expanded = normals.unsqueeze(1)     # [F, 1, 3]
                projections = torch.sum(faces_vertices_grad * normals_expanded, dim=2).abs()
                faces_grad = projections.sum(dim=1) 
                info['faces_grad'] = faces_grad
            else:
                info['faces_grad'] = None
            
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.wandb_on and cfg.log_every > 0 and step % cfg.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.run.log({
                    "train/loss": loss.item(),
                    "train/l1loss": l1loss.item(),
                    "train/ssimloss": ssimloss.item(),
                    "train/num_GS": len(self.splats["means"]),
                    "train/mem": mem}, step)
                if cfg.depth_loss:
                    self.run.log({"train/depthloss": depthloss.item()}, step)
                if cfg.normal_loss:
                    self.run.log({"train/normalloss": normalloss.item()}, step)
                if cfg.sky_mask_on:
                    self.run.log({"train/skyloss": skyloss.item()}, step)
                if cfg.depth_rank_loss:
                    self.run.log({"train/depthrankloss": depth_rank_loss.item()}, step)
                if cfg.plane_normal_map_loss:
                    self.run.log({"train/plane_normal_map_loss": plane_normal_map_loss.item()}, step)
                if cfg.dist_loss:
                    self.run.log({"train/distloss": distloss.item()}, step)
                if cfg.log_save_image and step % (cfg.log_every * 5) == 0:
                    canvas = torch.cat([pixels, colors[..., :3]], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.run.log({"train/render": wandb.Image(canvas, caption=f"step: {step}")})

            if step > 0 and step % 1000 == 0:
                # save .ply
                point_cloud_path = f"{self.ply_dir}/ply_{step:05d}_rank{self.world_rank}.ply"
                save_ply(point_cloud_path, self.splats)

                if cfg.bg_gs_start_iter <= step < cfg.bg_gs_stop_iter:
                    point_cloud_path = f"{self.ply_dir}/bg_ply_{step:05d}_rank{self.world_rank}.ply"
                    save_ply(point_cloud_path, self.bg_splats, is_bg=True)
            
                # save mesh
                vertices = self.mesh_params["vertices"].detach().cpu().numpy()
                faces = self.mesh_params["faces"].cpu().numpy()
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d.io.write_triangle_mesh(f"{self.ply_dir}/mesh_{step:05d}_rank{self.world_rank}.ply", mesh)
                if world_size > 1:
                    dist.barrier()

                    if world_rank == 0:
                        # merge ply of diffrent world ranks
                        merge_ply(self.ply_dir, step, self.world_size)

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:05d}.json", "w") as f:
                    json.dump(stats, f)

                data = {"step": step, "splats": self.splats}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()

                # save .pt
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt",
                )
                
                # save .ply
                point_cloud_path = f"{self.ply_dir}/ply_{step:05d}_rank{self.world_rank}.ply"
                save_ply(point_cloud_path, self.splats)

                vertices = self.mesh_params["vertices"].detach().cpu().numpy()
                faces = self.mesh_params["faces"].cpu().numpy()
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d.io.write_triangle_mesh(f"{self.ply_dir}/mesh_{step:05d}_rank{self.world_rank}.ply", mesh)
                
                if cfg.bg_gs_start_iter <= step < cfg.bg_gs_stop_iter:
                    point_cloud_path = f"{self.ply_dir}/bg_ply_{step:05d}_rank{self.world_rank}.ply"
                    save_ply(point_cloud_path, self.bg_splats, is_bg=True)
                if world_size > 1:
                    dist.barrier()

                    if world_rank == 0:
                        # merge ply of diffrent world ranks
                        merge_ply(self.ply_dir, step, self.world_size)

            if cfg.bg_gs_start_iter <= step < cfg.bg_gs_stop_iter:
                bg_info = {}
                for key in ["width", "height", "n_cameras"]:
                    bg_info[key] = info[key]
                num_gs_from_mesh = self.splats["means"].shape[0]
                bg_info["radii"] = info["radii"][:, num_gs_from_mesh:]
                info["radii"] = info["radii"][:, :num_gs_from_mesh]
                grad = info["gradient_2dgs"].grad.clone()
                bg_info["gradient_2dgs"] = info["gradient_2dgs"][:, num_gs_from_mesh:]
                info["gradient_2dgs"] = info["gradient_2dgs"][:, :num_gs_from_mesh]
                bg_info["gradient_2dgs"].grad = grad[:, num_gs_from_mesh:]
                info["gradient_2dgs"].grad = grad[:, :num_gs_from_mesh]

                self.bg_strategy.step_post_backward(
                    params=self.bg_splats,
                    optimizers=self.optimizers,
                    state=self.bg_strategy_state,
                    step=step,
                    info=bg_info,
                    packed=cfg.packed,
                    is_bg=True
                )
            
            # Densify
            self.strategy.step_post_backward(
                mesh_params=self.mesh_params,
                gs_params=self.splats,
                cfg=cfg,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mesh_state=self.mesh_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
            
            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
            torch.cuda.empty_cache()

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Remove degradation faces
            self.strategy.prune_degradation_faces(
                cfg=self.cfg,
                gs_params=self.splats,
                mesh_params=self.mesh_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mesh_state=self.mesh_state,
            )

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                # self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            val_image_ids = data["image_id"].to(device)
            val_image_ids = val_image_ids * (cfg.test_every - 1)
            
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                render_depths,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                val_image_ids=val_image_ids,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if world_rank == 0:
                # get valid mask
                mask = torch.ones((1, height, width, 1), dtype=torch.bool).to(device)
                if self.cfg.sky_mask_on:
                    smask = data["smask"].to(device)
                    mask = torch.logical_and(mask, torch.logical_not(smask))
                if self.cfg.dynamic_mask_on:
                    dmask = data["dmask"].to(device)
                    mask = torch.logical_and(mask, dmask)

                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
                )

                pixels = (pixels*mask).permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = (colors*mask).permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))

                # # write median depths
                # render_median = (render_median - render_median.min()) / (render_median.max() - render_median.min())
                # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
                # # render_median = render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_median_depth_{step}.png", (render_median * 255).astype(np.uint8)
                # )
        
                # write normals
                normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
                normals_output = (normals * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_normal_{step}.png", normals_output
                )

                # write normals from depth
                normals_from_depth *= alphas.squeeze(0).detach()
                normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
                normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                    np.max(normals_from_depth) - np.min(normals_from_depth)
                )
                normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
                if len(normals_from_depth_output.shape) == 4:
                    normals_from_depth_output = normals_from_depth_output.squeeze(0)
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                    normals_from_depth_output,
                )

                # write distortions
                render_dist = render_distort
                dist_max = torch.max(render_dist)
                dist_min = torch.min(render_dist)
                render_dist = (render_dist - dist_min) / (dist_max - dist_min)
                render_dist = (
                    colormap(render_dist.cpu().numpy()[0])
                    .permute((1, 2, 0))
                    .numpy()
                    .astype(np.uint8)
                )
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_distortions_{step}.png", render_dist
                )

        if world_rank == 0:
            ellipse_time /= len(valloader)

            psnr = torch.stack(metrics["psnr"]).mean()
            ssim = torch.stack(metrics["ssim"]).mean()
            lpips = torch.stack(metrics["lpips"]).mean()
            print(
                f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
            # save stats as json
            stats = {
                "psnr": psnr.item(),
                "ssim": ssim.item(),
                "lpips": lpips.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to wandb
            if self.cfg.wandb_on:
                self.run.log({f"val/{k}": v for k, v in stats.items()}, step)

    @torch.no_grad()
    def render(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[:]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(len(camtoworlds), desc="Rendering"):
            renders, _, normals, _, _, render_depths, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]

            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0).cpu().numpy()  # [H, W, 3]
            depths = render_depths.squeeze()  # [H, W, 1]
            normals = torch.nn.functional.normalize(normals[0], dim=2)
            normals = normals.cpu().numpy()
            
            save_image(colors, f"{self.render_dir}/render_{i:04d}.png")
            depth_gray2rgb(depths, f"{self.render_dir}/depth_{i:04d}.png", 0.1, 5)
            normal_rgb2sn(normals, f"{self.render_dir}/normal_{i:04d}.png")

    @torch.no_grad()
    def export_mesh(self, depth_trunc=3, mesh_res=1024):
        voxel_size = depth_trunc / mesh_res
        sdf_trunc = 5.0 * voxel_size
        
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[:]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(len(camtoworlds), desc="Exporting mesh"):
            renders, _, normals, _, _, render_depths, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]          
            rgb = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depth = render_depths.squeeze()  # [H, W]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            # ndc2pix = torch.tensor([
            #     [width / 2, 0, 0, (width-1) / 2],
            #     [0, height / 2, 0, (height-1) / 2],
            #     [0, 0, 0, 1]]).float().cuda().T
            # intrins =  (K @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height,
                cx = K[0,2].item(),
                cy = K[1,2].item(), 
                fx = K[0,0].item(), 
                fy = K[1,1].item()
            )
            extrinsic=np.linalg.inv(np.asarray((camtoworlds[i]).cpu().numpy()))
            volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=extrinsic)
            del renders, normals, rgb, depth
            gc.collect()
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join(self.cfg.result_dir, "fuse.ply"), mesh)
        print("mesh saved at {}".format(os.path.join(self.cfg.result_dir, "fuse.ply")))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
        o3d.io.write_triangle_mesh(os.path.join(self.cfg.result_dir, "fuse_post.ply"), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(self.cfg.result_dir, "fuse_post.ply")))

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    run = wandb.init(
        project="test_4gpu",
        group="DDP",  # all runs for the experiment in one group
    ) if world_rank == 0 and cfg.wandb_on else None
    
    runner = Runner(local_rank, world_rank, world_size, cfg, run)

    if cfg.ckpt is not None:
        # run eval only
        if world_rank == 0:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=True)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            # runner.eval(step=ckpts[0]["step"])
            # runner.render_traj(step=ckpts[0]["step"])
            runner.export_mesh()
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    wandb.setup()
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
