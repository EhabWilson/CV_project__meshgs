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
from utils import (AppearanceOptModule, CameraOptModule, knn, 
                   rgb_to_sh, set_random_seed, colormap, apply_depth_colormap, 
                   l1_loss, focal_l1_loss, tv_loss, sky_loss, depth_ranking_loss,
                   save_ply, merge_ply, load_ply,
                   save_image, depth_gray2rgb, normal_rgb2sn, post_process_mesh, ssim, 
                   least_squares, sobel_edge, dice_loss, consistency_loss, compute_quats)

from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import DefaultStrategy
from matplotlib import pyplot as plt

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[List[str]] = None
    # Path to the gs .ply file. If provide, it will skip training and render a video
    gs_ply: Optional[str] = None

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
    # Load instance map
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
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

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

    # Mono depth loss
    mono_depth_loss: bool = False
    # Iteration to start mono_depth_regularization
    mono_depth_start_iter: int = 0
    # Weight for mono depth loss
    mono_depth_lambda: float = 0.5
    # Depth rank mode
    depth_rank_mode: Literal["random_sample", "global"] = "global"
    # Depth rank loss
    depth_rank_loss: bool = False
    # Weight for depth rank loss
    depth_rank_lambda: float = 1
    # Num of sampled point pairs
    num_sampled_pairs: int = 4096*4
    # Depth smooth loss
    depth_smooth_loss: bool = False
    # Weight for depth smooth loss
    depth_smooth_lambda: float = 1e-1

    # Load plane masks
    plane_consistency_loss: bool = False
    # Weight for plane normal consistency
    plane_consistency_lambda: float = 1e-1
    # Load normal map
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
    # Wandb project name
    wandb_project: str = "test"
    # Wandb group name
    wandb_group: str = "DDP"
    # How often to dump information to wandb
    log_every: int = 100
    # Save training images to wandb
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
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        normals = parser.points_normal if parser.points_normal is not None else None
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
        normals = None
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")
    # Initialize the GS size to be the average dist of the 3 nearest neighbors
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
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

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
            load_plane_masks=cfg.plane_consistency_loss or cfg.plane_normal_map_loss,
            load_normal_maps=cfg.plane_normal_map_loss,
            instance_map_on=cfg.depth_smooth_loss or cfg.instance_map_on
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
        self.splats, self.optimizers = create_splats_with_optimizers(
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
        )
        # DEBUG_TMP
        print("Saving initial Gaussians")
        save_ply(os.path.join(self.cfg.result_dir,"plys/init.ply"),self.splats)
        
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        assert cfg.reset_every >= cfg.pause_refine_after_reset, "[ERROR] reset_every must be >= pause_refine_after_reset."
        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            pause_refine_after_reset=cfg.pause_refine_after_reset,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient="gradient_2dgs",
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(
            scene_scale = self.scene_scale
        )

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
            if self.distributed:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if self.distributed:
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
            if self.distributed:
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
        instance_map: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

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
            instance_map=instance_map,
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
                instance_id_of_gaussians,
            ) = renders
        elif self.model_type == "2dgs-inria":
            render_colors, render_alphas = renders
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
            instance_id_of_gaussians,
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
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
            ),
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
        pbar = tqdm.tqdm(range(init_step, max_steps), ncols=80)
        for step in pbar:
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
            if cfg.depth_smooth_loss or cfg.instance_map_on:
                imap = data["imap"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            (
                renders,
                alphas,
                render_normals,
                render_normals_from_depth,
                render_distort,
                render_median,
                render_depths,
                instance_id_of_gaussians,
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
                distloss=cfg.dist_loss,
                instance_map=torch.zeros((1, height, width, 1), device=device).int()
            )
            if data["image_name"][0][-4:] in ["0182", "0432", "0757", "0999"]:
                render_depths_ = (render_depths - render_depths.min()) / (render_depths.max() - render_depths.min())
                # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
                render_depths_ = render_depths_.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
                imageio.imwrite(
                    f"{self.eval_dir}/val_{data['image_name'][0][-4:]}_depth_{step}.png", (render_depths_ * 255).astype(np.uint8)
                )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # get valid mask
            h, w = pixels.shape[1:3]
            mask = torch.ones((1, h, w, 1), dtype=torch.bool).to(device)

            if cfg.sky_mask_on:
                smask = data["smask"].to(device)
                mask = torch.logical_and(mask, torch.logical_not(smask))
            if cfg.dynamic_mask_on:
                dmask = data["dmask"].to(device)
                mask = torch.logical_and(mask, torch.logical_not(dmask))
            
            # get mono depth
            if cfg.mono_depth_loss or cfg.depth_rank_loss:
                mono_depths = data["mono_depths"].to(device)

            # get plane masks
            if self.cfg.plane_consistency_loss or self.cfg.plane_normal_map_loss:
                plane_mask = data["plane_mask"].to(device)

            if self.cfg.plane_normal_map_loss:
                normal_map = data["normal_map"].to(device)

            # loss
            if cfg.focal_loss_on:
                l1loss = focal_l1_loss(colors, pixels, mask)
            else:
                l1loss = l1_loss(colors, pixels, mask)
            
            ssimloss = 1.0 - ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), mask=mask)
            
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            
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
                    render_depths_anchor = render_depths.squeeze(-1)[:,1:-1,1:-1][:,::2,::2]
                    render_depths_h_shift = render_depths.squeeze(-1)[:,:,1:-1][:,::2,::2]
                    render_depths_w_shift = render_depths.squeeze(-1)[:,1:-1,][:,::2,::2]
                    mono_depths_anchor = mono_depths.squeeze(-1)[:,1:-1,1:-1][:,::2,::2]
                    mono_depths_h_shift = mono_depths.squeeze(-1)[:,:,1:-1][:,::2,::2]
                    mono_depths_w_shift = mono_depths.squeeze(-1)[:,1:-1,][:,::2,::2]
                    depths_rank_top = torch.where(mono_depths_h_shift[:,:-1,:]>mono_depths_anchor, 1, -1)
                    depths_rank_bottom = torch.where(mono_depths_h_shift[:,1:,:]>mono_depths_anchor, 1, -1)
                    depths_rank_left = torch.where(mono_depths_w_shift[:,:,:-1]>mono_depths_anchor, 1, -1)
                    depths_rank_right = torch.where(mono_depths_w_shift[:,:,1:]>mono_depths_anchor, 1, -1)
                    depth_rank_loss = torch.mean(torch.maximum(
                        torch.cat([
                            depths_rank_top * (render_depths_h_shift[:,:-1,:] - render_depths_anchor) - margin,
                            depths_rank_bottom * (render_depths_h_shift[:,1:,:] - render_depths_anchor) - margin,
                            depths_rank_left * (render_depths_w_shift[:,:,:-1] - render_depths_anchor) - margin,
                            depths_rank_right * (render_depths_w_shift[:,:,1:] - render_depths_anchor) - margin,
                        ]), torch.tensor(0)
                    ))
                loss += depth_rank_loss * curr_depth_rank_lambda

            # plane consistency
            plane_consistency_loss = torch.tensor(0.0, device=device)
            if self.cfg.plane_consistency_loss and step >= 3000:
                normals = render_normals.clone().squeeze(0)     # [H, W, 3]
                plane_mask = plane_mask.squeeze(0)              # [H, W]
                plane_ids = torch.unique(plane_mask)[1:]        # [N]        
                
                if plane_ids.size(0) > 0:
                    # Expand normals and masks
                    plane_masks = torch.stack([(plane_mask == plane_id).float() for plane_id in plane_ids], dim=0)  # [N, H, W]
                    plane_masks_expanded = plane_masks.unsqueeze(-1)                                               # [N, H, W, 1]
                    normals_expanded = normals.unsqueeze(0).expand(plane_masks.size(0), -1, -1, -1)                 # [N, H, W, 3]

                    # Select normals for planes
                    normals_selected = normals_expanded * plane_masks_expanded                                      # [N, H, W, 3]
                    normals_selected_flatten = normals_selected.view(plane_masks.size(0), -1, 3)                    # [N, H*W, 3]
                    plane_masks_flatten = plane_masks.view(plane_masks.size(0), -1)                                 # [N, H*W]
                    normals_filtered = [normals_selected_flatten[i, plane_masks_flatten[i] > 0] for i in range(plane_masks.size(0))]

                    # Calculate variance
                    variance_per_plane = torch.stack([torch.var(norm, dim=0) for norm in normals_filtered], dim=0)  # [N, 3]

                    # Area-Weighted
                    plane_areas = torch.sum(plane_masks_flatten > 0, dim=1).float()                                 # [N]
                    weights = plane_areas / torch.sum(plane_areas)                                                  # [N]

                    loss_per_plane = torch.sum(variance_per_plane, dim=1) * weights                                 # [N]

                    plane_consistency_loss = torch.mean(loss_per_plane) * cfg.plane_consistency_lambda

                loss += plane_consistency_loss 

            # plane normals with pseudo normals
            plane_normal_map_loss = torch.tensor(0.0, device=device)
            if self.cfg.plane_normal_map_loss and step >= self.cfg.plane_normal_map_loss_start_iter:
                normal_map = normal_map @ camtoworlds[0, :3, :3].T      # Transform to world coordinate
                normal_map = normal_map.squeeze(0)                      # [H, W, 3]

                normals = render_normals.clone().squeeze(0)             # [H, W, 3]: world coordinate
                
                plane_mask = plane_mask.squeeze(0)                      # [H, W]
                plane_ids = torch.unique(plane_mask)[1:]                # [N]
                
                plane_normal_losses = []
                for plane_id in plane_ids:
                    plane_mask_curr = (plane_mask == plane_id).float()  # [H, W]

                    # Expand mask and apply to normal_map and normals
                    plane_mask_expanded = plane_mask_curr.unsqueeze(-1)                       # [H, W, 1]
                    normal_map_selected = normal_map * plane_mask_expanded                    # [H, W, 3]
                    normals_selected = normals * plane_mask_expanded                          # [H, W, 3]

                    # Flatten and filter valid points
                    normal_map_flatten = normal_map_selected.view(-1, 3)                      # [H*W, 3]
                    normals_flatten = normals_selected.view(-1, 3)                            # [H*W, 3]
                    plane_mask_flatten = plane_mask_curr.view(-1)                             # [H*W]

                    # Filter valid points
                    normal_map_filtered = normal_map_flatten[plane_mask_flatten > 0]          # [L, 3]
                    normals_filtered = normals_flatten[plane_mask_flatten > 0]                # [L, 3]

                    if normals_filtered.shape[0] > 0 and normal_map_filtered.shape[0] > 0:
                        plane_normal_loss = torch.abs(normals_filtered - normal_map_filtered).sum(dim=1)
                        plane_normal_loss = torch.clamp(plane_normal_loss, min=0.05).mean()

                        plane_normal_losses.append(plane_normal_loss)
                # Calculate overall loss
                if len(plane_normal_losses) > 0:
                    plane_normal_map_loss = torch.mean(torch.stack(plane_normal_losses)) * cfg.plane_normal_map_lambda

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
            
            if cfg.mono_depth_loss:
                if step > cfg.mono_depth_start_iter:
                    curr_mono_depth_lambda = cfg.mono_depth_lambda
                else:
                    curr_mono_depth_lambda = 0.0
                b, h, w, _ = mono_depths.shape
                # mono_depths : [B,H,W,1]
                # render_depths : [B,H,W,1]

                # mask out dynamic and sky
                mono_depths = mono_depths.squeeze(-1).reshape(-1, h * w)[mask.reshape(-1 , h * w)].reshape(b, -1)
                render_depths = render_depths.squeeze(-1).reshape(-1, h * w)[mask.reshape(-1, h * w)].reshape(b,-1)
                
                # remove far depths
                mono_depths_threshold = torch.quantile(mono_depths, 0.1, dim=-1, keepdim=True)
                mono_depths_mask = mono_depths >= mono_depths_threshold
                render_depths_threshold = torch.quantile(render_depths, 0.9, dim=-1, keepdim=True)
                render_depths_mask = mono_depths <= render_depths_threshold

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
                    h_r = np.random.randint(h-box_h+1, size=(cfg.num_sampled_pairs, 1))
                    w_r = np.random.randint(w-box_w+1, size=(cfg.num_sampled_pairs, 1))
                    h_r_delta = np.random.randint(box_h, size=(cfg.num_sampled_pairs, 2))
                    w_r_delta = np.random.randint(box_w, size=(cfg.num_sampled_pairs, 2))
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
            
            if cfg.depth_smooth_loss:
                if step > cfg.mono_depth_start_iter:
                    curr_depth_smooth_lambda = cfg.depth_smooth_lambda
                else:
                    curr_depth_smooth_lambda = 0.0
                box_h = 10
                box_w = 10
                margin = 1e-4
                # sample center of several patches
                h_r = np.random.randint(box_h//2, h-box_h//2+1, size=(cfg.num_sampled_pairs, 1))
                w_r = np.random.randint(box_w//2, w-box_w//2+1, size=(cfg.num_sampled_pairs, 1))
                # sample two points from the same instance for each patch
                h_r_delta = np.random.randint(-box_h//2, box_h//2, size=(cfg.num_sampled_pairs, 2))
                w_r_delta = np.random.randint(-box_w//2, box_w//2, size=(cfg.num_sampled_pairs, 2))
                h_r = np.concatenate((h_r+h_r_delta[:,:1],h_r+h_r_delta[:,1:2]), axis=1)
                w_r = np.concatenate((w_r+w_r_delta[:,:1],w_r+w_r_delta[:,1:2]), axis=1)
                render_depths_0 = render_depths.reshape(-1, h*w)[:, h_r[:,0]*w+w_r[:,0]].reshape(-1)
                render_depths_1 = render_depths.reshape(-1, h*w)[:, h_r[:,1]*w+w_r[:,1]].reshape(-1)
                imap_0 = imap.reshape(-1, h*w)[:, h_r[:,0]*w+w_r[:,0]].reshape(-1)
                imap_1 = imap.reshape(-1, h*w)[:, h_r[:,1]*w+w_r[:,1]].reshape(-1)
                depth_smooth_loss = torch.mean(torch.maximum(
                    torch.abs(render_depths_0 - render_depths_1)[imap_0 == imap_1] - margin, torch.tensor(0)
                ))
                loss += depth_smooth_loss * curr_depth_smooth_lambda

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
                # total normal loss
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
            
            if cfg.edge_loss:
                sobel_render = sobel_edge(colors)
                sobel_gt = sobel_edge(pixels)
                edge_diceloss = dice_loss(sobel_render, sobel_gt, mask=mask) * cfg.edge_lambda
                loss += edge_diceloss 

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.plane_consistency_loss:
                desc += f"plane consistency loss={plane_consistency_loss.item():.6f}| "
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
                if cfg.mono_depth_loss:
                    self.run.log({"train/monodepthloss": pdepth_loss.item()}, step)
                if cfg.depth_rank_loss:
                    self.run.log({"train/depthrankloss": depth_rank_loss.item()}, step)
                if cfg.depth_smooth_loss:
                    self.run.log({"train/depthsmoothloss": depth_smooth_loss.item()}, step)
                if cfg.edge_loss:
                    self.run.log({"train/edgeloss": edge_diceloss.item()}, step)
                if cfg.sky_mask_on:
                    self.run.log({"train/skyloss": skyloss.item()}, step)
                if cfg.depth_rank_loss:
                    self.run.log({"train/depthrankloss": depth_rank_loss.item()}, step)
                if cfg.plane_consistency_loss:
                    self.run.log({"train/planeconsistencyloss": plane_consistency_loss.item()}, step)
                if cfg.plane_normal_map_loss:
                    self.run.log({"train/plane_normal_map_loss": plane_normal_map_loss.item()}, step)
                if cfg.dist_loss:
                    self.run.log({"train/distloss": distloss.item()}, step)
                if cfg.log_save_image and step % (cfg.log_every * 5) == 0:
                    canvas = torch.cat([pixels, colors[..., :3]], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.run.log({"train/render": wandb.Image(canvas, caption=f"step: {step}")})

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                ckpt = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if self.distributed:
                        ckpt["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        ckpt["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if self.distributed:
                        ckpt["app_module"] = self.app_module.module.state_dict()
                    else:
                        ckpt["app_module"] = self.app_module.state_dict()

                # save .pt
                torch.save(
                    ckpt, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt",
                )
                
                # save .ply
                point_cloud_path = f"{self.ply_dir}/ply_{step}_rank{self.world_rank}.ply"
                save_ply(point_cloud_path, self.splats)
                if world_size > 1:
                    dist.barrier()

                if self.distributed:
                    dist.barrier()
                if world_rank == 0:
                    # merge ply of diffrent world ranks
                    merge_ply(self.ply_dir, step, self.world_size)
            
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
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

            torch.cuda.synchronize()
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
                    mask = torch.logical_and(mask, torch.logical_not(dmask))

                # # write images
                # canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
                # )

                pixels = (pixels*mask).permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = (colors*mask).permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))

                # # DEBUG_TMP write depths
                # render_depths = render_depths.detach().cpu().squeeze(0)
                # np.save(f"{self.eval_dir}/val_{i:04d}_depths_{step}.npy", render_depths.numpy())

                # # write median depths
                # render_median = (render_median - render_median.min()) / (render_median.max() - render_median.min())
                # # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
                # render_median = render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_median_depth_{step}.png", (render_median * 255).astype(np.uint8)
                # )
        
                # # write normals
                # normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
                # normals_output = (normals * 255).astype(np.uint8)
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_normal_{step}.png", normals_output
                # )

                # # write normals from depth
                # normals_from_depth *= alphas.squeeze(0).detach()
                # normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
                # normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                #     np.max(normals_from_depth) - np.min(normals_from_depth)
                # )
                # normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
                # if len(normals_from_depth_output.shape) == 4:
                #     normals_from_depth_output = normals_from_depth_output.squeeze(0)
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                #     normals_from_depth_output,
                # )

                # # write distortions
                # render_dist = render_distort
                # dist_max = torch.max(render_dist)
                # dist_min = torch.min(render_dist)
                # render_dist = (render_dist - dist_min) / (dist_max - dist_min)
                # render_dist = (
                #     colormap(render_dist.cpu().numpy()[0])
                #     .permute((1, 2, 0))
                #     .numpy()
                #     .astype(np.uint8)
                # )
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_distortions_{step}.png", render_dist
                # )

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
            renders, _, normals, _, _, _, _, _ = self.rasterize_splats(
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
            depths = renders[0, ..., 3:4].squeeze()  # [H, W, 1]
            normals = torch.nn.functional.normalize(normals[0], dim=2)
            normals = normals.cpu().numpy()
            
            save_image(colors, f"{self.render_dir}/render_{i:04d}.png")
            depth_gray2rgb(depths, f"{self.render_dir}/depth_{i:04d}.png", 0.1, 5)
            normal_rgb2sn(normals, f"{self.render_dir}/normal_{i:04d}.png")

    @torch.no_grad()
    def export_mesh(self, depth_trunc=5, mesh_res=1024):
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
            renders, _, normals, _, _, _, _, _ = self.rasterize_splats(
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
            depth = renders[0, ..., 3:4]  # [H, W]

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

        render_colors, _, _, _, _, _, _ = self.rasterize_splats(
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
        project=cfg.wandb_project,
        group=cfg.wandb_group,  # all runs for the experiment in one group
        config={'scene': cfg.data_dir, 'depth_rank_lambda': cfg.depth_rank_lambda, 'depth_rank_batch': cfg.num_sampled_pairs}
    ) if world_rank == 0 and cfg.wandb_on else None
    
    runner = Runner(local_rank, world_rank, world_size, cfg, run)

    if cfg.gs_ply is not None or cfg.ckpt is not None:
        # render images and export mesh
        if world_rank == 0:
            if cfg.gs_ply is not None:
                load_ply(cfg.gs_ply, runner.splats)
            elif cfg.ckpt is not None:
                ckpts = [
                    torch.load(file, map_location=runner.device, weights_only=True)
                    for file in cfg.ckpt
                ]
                for k in runner.splats.keys():
                    runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            else:
                raise NotImplementedError
            # runner.render(step=ckpts[0]["step"])
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
