import random
import cv2
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree, ConvexHull
import trimesh
from tqdm import tqdm
from torch import nn, Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps
from plyfile import PlyData, PlyElement
import os
import open3d as o3d
from math import exp
import math
import struct
from typing import Union
from pytorch3d.transforms import quaternion_apply
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
import struct

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, val_image_ids : Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is not None:
            # train
            embeds = self.embeds(embed_ids)  # [C, D2]
        elif val_image_ids is not None:
            # eval
            embed_prev = self.embeds(val_image_ids - 1) if val_image_ids - 1 >=0 else torch.zeros(C, self.embed_dim, device=features.device)
            embed_next = self.embeds(val_image_ids) if val_image_ids < self.embeds.weight.shape[0] else torch.zeros(C, self.embed_dim, device=features.device)
            
            embeds = embed_prev + embed_next if torch.all(embed_prev == 0) or torch.all(embed_next == 0) else (embed_prev + embed_next) / 2
        else:
            assert False
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        
        # if embed_ids is None:
        #     embeds = torch.zeros(C, self.embed_dim, device=features.device)
        # else:
        #     embeds = self.embeds(embed_ids)  # [C, D2]
        
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img

def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]

def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def l1_loss(img1, img2, mask):
    loss = torch.abs((img1 - img2)) * mask
    return loss.mean()

def focal_l1_loss(img1, img2, mask=None, alpha=2.0, beta=2.0):
    error = torch.abs(img1 - img2)

    focal_weight = alpha * (1 - torch.exp(-beta * error))
    
    focal_error = focal_weight * error * mask

    return focal_error.mean()

def tv_loss(normals, mask):
    h_diff = torch.abs(normals[:, :-1, :] - normals[:, 1:, :])
    h_diff = h_diff * mask[:, 1:, :]
    w_diff = torch.abs(normals[:, :, :-1] - normals[:, :, 1:])
    w_diff = w_diff * mask[:, :, 1:]

    return h_diff.mean() + w_diff.mean()

def sky_loss(alphas, sky_mask):
    skyloss = F.binary_cross_entropy(alphas, 1-sky_mask.float(), reduction="none")
    return skyloss.mean()

def save_image(image, output_file):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open(output_file, 'wb') as f:
        image = Image.fromarray(
            (np.clip(image, 0., 1.)*255).astype(np.uint8))
        image.save(f, 'PNG')
    
def depth_gray2rgb(depth, output_file, min_depth=0.1, max_depth=40.0):
    # depth = apply_depth_colormap(depth, near_plane=min_depth, far_plane=max_depth)
    depth = depth.cpu().numpy()
    depth = np.clip(depth, min_depth, max_depth)
    depth = (depth - min_depth) / (max_depth - min_depth) * 255
    depth = depth.astype(np.uint8)
    depth = (colormaps["Spectral_r"](depth)[:,:,:3]*255).astype(np.uint8)
    image = Image.fromarray(depth, "RGB")
    image.save(output_file)

def normal_rgb2sn(normals, output_file):
    # 交换第1和第3分量
    a = np.copy(normals[:,:,0])
    b = np.copy(normals[:,:,2])
    normals[:,:,0] = b
    normals[:,:,2] = a
    # 反转法线方向
    normals = normals @ np.diag([-1, -1, -1])

    normals = (normals + 1) / 2.0 * 255.0
    normals = normals.astype(np.uint8)

    cv2.imwrite(output_file, normals)

# Experimental
def construct_list_of_attributes(splats, is_bg=False):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    names = ["sh0", "shN", "scales", "quats"]
    if is_bg:
        names = ["bg_"+name for name in names]
    for i in range(splats[names[0]].shape[1]*splats[names[0]].shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(splats[names[1]].shape[1]*splats[names[1]].shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    # ! 2D GS scale
    for i in range(splats[names[2]].shape[1] - 1):
        l.append('scale_{}'.format(i))
    for i in range(splats[names[3]].shape[1]):
        l.append('rot_{}'.format(i))
    return l

# Experimental
@torch.no_grad()
def save_ply(path, splats, is_2dgs=True, is_bg=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # FIXME
    if "sh0" not in splats.keys():
        return

    names = ["means", "sh0", "shN", "opacities", "scales", "quats"]
    if is_bg:
        names = ["bg_"+name for name in names]

    xyz = splats[names[0]].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = splats[names[1]].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = splats[names[2]].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats[names[3]].detach().unsqueeze(-1).cpu().numpy()
    scale = splats[names[4]].detach().cpu().numpy()
    if is_2dgs:
        scale = scale[:, :2]
    rotation = splats[names[5]].detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats, is_bg)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

# Experimental
@torch.no_grad()
def load_ply(path, splats, is_2dgs=True):
    plydata = PlyData.read(path)
    N = len(plydata.elements[0])
    means = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])
    quats = np.stack((np.asarray(plydata.elements[0]["rot_0"]),
                    np.asarray(plydata.elements[0]["rot_1"]),
                    np.asarray(plydata.elements[0]["rot_2"]),
                    np.asarray(plydata.elements[0]["rot_3"])),  axis=1)
    property_names = plydata['vertex'].data.dtype.names

    scale_property_names = [name for name in property_names if name.startswith('scale_')]
    scale_property_names.sort(key=lambda x: int(x.split('_')[-1]))
    scale_arrays = [plydata['vertex'].data[name] for name in scale_property_names]
    scales = np.column_stack(scale_arrays)

    f_dc_property_names = [name for name in property_names if name.startswith('f_dc_')]
    f_dc_property_names.sort(key=lambda x: int(x.split('_')[-1]))
    f_dc_arrays = [plydata['vertex'].data[name] for name in f_dc_property_names]
    sh0 = np.column_stack(f_dc_arrays).reshape(N, 1, 3)

    f_rest_property_names = [name for name in property_names if name.startswith('f_rest_')]
    f_rest_property_names.sort(key=lambda x: int(x.split('_')[-1]))
    f_rest_arrays = [plydata['vertex'].data[name] for name in f_rest_property_names]
    shN = np.column_stack(f_rest_arrays).reshape(N, -1, 3)

    scales_ref = splats['scales'].detach().cpu().numpy()
    scales_tmp = np.zeros((N, scales_ref.shape[-1]), dtype=scales_ref.dtype)
    scales_tmp[:,:scales.shape[-1]] = scales # handle 2DGS
    scales = scales_tmp
    data_dict = {"means":means, "opacities":opacities, "quats":quats, "scales":scales, "sh0":sh0, "shN":shN, }

    for k in splats.keys():
        splats[k] = nn.Parameter(torch.tensor(data_dict[k], dtype=splats[k].dtype, device=splats[k].device).requires_grad_(splats[k].requires_grad))

@torch.no_grad()
def merge_ply(path, step, world_size):
    ply_paths = [f"{path}/ply_{step}_rank{world_rank}.ply" for world_rank in range(world_size)]
    
    merged_elements = []

    for ply_path in ply_paths:
        plydata = PlyData.read(ply_path)
        
        merged_elements.append(plydata.elements[0].data)

    merged_elements = np.concatenate(merged_elements)

    el = PlyElement.describe(merged_elements, 'vertex')
    PlyData([el]).write(f"{path}/ply_{step}.ply")

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask=mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    b, c, h, w = ssim_map.shape
    ssim_map = ssim_map.reshape(b, c, h*w)
    if mask is not None:
        ssim_map = ssim_map[:, :, mask.reshape(-1)]
     
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(2).mean(1).mean(0)

def least_squares(img1, img2):
    B, L = img1.shape
    ones = torch.ones(B, L, device=img1.device)
    X = torch.stack([img1, ones], dim=-1)  # [B, L, 2]

    solution = torch.linalg.lstsq(X, img2.unsqueeze(-1)).solution  # [B, 2, 1]

    return solution.squeeze(-1)  # [B, 2]

def sobel_edge(image):
    """
    @input
    - image : [B, H, W, C]

    @output
    - sobel_result : [B, H, W, 1]
    """
    gray = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :, 2]
    gray = gray.unsqueeze(1)  # Add channel dimension [B, 1, H, W]

    sobel_x = torch.tensor([[[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]], dtype=torch.float32).unsqueeze(0).to(image.device)
    sobel_y = torch.tensor([[[-1., -2., -1.],
                            [0.,  0.,  0.],
                            [1.,  2.,  1.]]], dtype=torch.float32).unsqueeze(0).to(image.device)
    
    grad_x = F.conv2d(gray, sobel_x.expand(1, 1, 3, 3), padding=1)
    grad_y = F.conv2d(gray, sobel_y.expand(1, 1, 3, 3), padding=1)

    # Compute magnitude of gradient
    sobel_result = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    return sobel_result.permute(0, 2, 3, 1)  # [B, H, W, 1]

def dice_loss(pred, target,mask, epsilon=1e-6):
    # pred and target are expected to have shape [B, H, W, 1]
    # Flatten each sample in the batch
    B, H, W, _ = pred.shape
    pred = (pred * mask).view(B, -1)
    target = (target * mask).view(B, -1)

    # Compute Dice coefficient for each sample in the batch
    intersection = (pred * target).sum(dim=1)
    dice_coef = (2. * intersection + epsilon) / (pred.sum(dim=1) + target.sum(dim=1) + epsilon)

    # Return mean Dice loss over the batch
    return 1 - dice_coef.mean()

def consistency_loss(image, masks):
    """
    
    @input
    - image [H, W, 3]
    - masks [N, H, W]

    @output
    - loss: Total Variation loss
    """
    N, H, W = masks.shape
    loss = 0.0
    
    for i in range(N):
        mask = masks[i]
        mask = mask.unsqueeze(-1) 
                
        diff_x = torch.abs(image[:, 1:, :] - image[:, :-1, :])  # [H, W-1, 3]
        diff_x = diff_x * mask[:, 1:, :] * mask[:, :-1, :]
        
        diff_y = torch.abs(image[1:, :, :] - image[:-1, :, :])  # [H-1, W, 3]
        diff_y = diff_y * mask[1:, :, :] * mask[:-1, :, :]

        mask_x_valid = mask[:, 1:, :].sum()
        mask_y_valid = mask[1:, :, :].sum()
        
        if mask_x_valid > 0:
            loss += diff_x.sum() / mask_x_valid
        if mask_y_valid > 0:
            loss += diff_y.sum() / mask_y_valid       

    return loss / N

def depth_ranking_loss(render_depths, mono_depths, kernel_size=3, dilation=1, margin = 1e-4):
    B, H, W, _ = render_depths.shape
    pad_size = kernel_size // 2 * dilation

    # pad tensor before unfolding
    padded_render_depths = F.pad(render_depths.squeeze(-1), (pad_size, pad_size, pad_size, pad_size), mode='replicate') # (B, H+K//2*2, W+K//2*2)
    
    padded_mono_depths = F.pad(mono_depths.squeeze(-1), (pad_size, pad_size, pad_size, pad_size), mode='replicate') # (B, H+K//2*2, W+K//2*2)

    # unfold tensor to (B, K*K, H, W)
    unfold_render_depths = F.unfold(padded_render_depths.squeeze(1), kernel_size=kernel_size, dilation=dilation)    # (B, K*K, H*W)
    unfold_render_depths = unfold_render_depths.view(B, kernel_size * kernel_size, H, W)    # (B, K*K, H， W)

    unfold_mono_depths = F.unfold(padded_mono_depths.squeeze(1), kernel_size=kernel_size, dilation=dilation)    # (B, K*K, H*W)
    unfold_mono_depths = unfold_mono_depths.view(B, kernel_size * kernel_size, H, W)    # (B, K*K, H， W)

    # get depth rank from mono depths
    depths_rank = torch.where(unfold_mono_depths > mono_depths.view(B, 1, H, W), 1, -1)

    # get dense depth ranking loss
    loss = torch.maximum(depths_rank * (unfold_render_depths - render_depths.view(B, 1, H, W)) - margin, torch.tensor(0)).mean()

    return loss

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrices to quaternions.
    R: [B, 3, 3]
    Returns quaternions: [B, 4] in (qw, qx, qy, qz) format
    """
    B = R.shape[0]
    q = torch.zeros((B, 4), device=R.device, dtype=R.dtype)

    m = R  # [B, 3, 3]
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    cond = trace > 0
    s = torch.zeros(B, device=R.device, dtype=R.dtype)

    s[cond] = torch.sqrt(trace[cond] + 1.0) * 2  # s = 4 * qw
    q[cond, 0] = 0.25 * s[cond]
    q[cond, 1] = (m[cond, 2, 1] - m[cond, 1, 2]) / s[cond]
    q[cond, 2] = (m[cond, 0, 2] - m[cond, 2, 0]) / s[cond]
    q[cond, 3] = (m[cond, 1, 0] - m[cond, 0, 1]) / s[cond]

    not_cond = ~cond
    m00 = m[not_cond, 0, 0]
    m11 = m[not_cond, 1, 1]
    m22 = m[not_cond, 2, 2]

    idx = torch.zeros(not_cond.sum(), dtype=torch.long, device=R.device)
    cond1 = (m00 > m11) & (m00 > m22)
    idx[cond1] = 0
    cond2 = (m11 > m22) & (~cond1)
    idx[cond2] = 1
    cond3 = (~cond1) & (~cond2)
    idx[cond3] = 2

    m01 = m[not_cond, 0, 1]
    m10 = m[not_cond, 1, 0]
    m02 = m[not_cond, 0, 2]
    m20 = m[not_cond, 2, 0]
    m12 = m[not_cond, 1, 2]
    m21 = m[not_cond, 2, 1]

    qn = q[not_cond]

    idx0 = idx == 0
    s0 = torch.sqrt(1.0 + m00[idx0] - m11[idx0] - m22[idx0]) * 2
    qn[idx0, 0] = (m21[idx0] - m12[idx0]) / s0
    qn[idx0, 1] = 0.25 * s0
    qn[idx0, 2] = (m01[idx0] + m10[idx0]) / s0
    qn[idx0, 3] = (m02[idx0] + m20[idx0]) / s0

    idx1 = idx == 1
    s1 = torch.sqrt(1.0 + m11[idx1] - m00[idx1] - m22[idx1]) * 2
    qn[idx1, 0] = (m02[idx1] - m20[idx1]) / s1
    qn[idx1, 1] = (m01[idx1] + m10[idx1]) / s1
    qn[idx1, 2] = 0.25 * s1
    qn[idx1, 3] = (m12[idx1] + m21[idx1]) / s1

    idx2 = idx == 2
    s2 = torch.sqrt(1.0 + m22[idx2] - m00[idx2] - m11[idx2]) * 2
    qn[idx2, 0] = (m10[idx2] - m01[idx2]) / s2
    qn[idx2, 1] = (m02[idx2] + m20[idx2]) / s2
    qn[idx2, 2] = (m12[idx2] + m21[idx2]) / s2
    qn[idx2, 3] = 0.25 * s2

    q[not_cond] = qn
    q = q / q.norm(dim=1, keepdim=True)
    return q

def triangles_to_ellipses(triangles):
    """
    @Params:
    - triangles: [F, 3, 3]

    @Returns:
    - centers: [F, 3, 3]
    - scales: [F, 3, 2]
    - quaternions: [F, 3, 4]
    """
    B = triangles.shape[0]

    device = triangles.device
    dtype = torch.float64
    triangles = triangles.to(dtype)

    A_std_2d = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    B_std_2d = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    C_std_2d = torch.tensor([0.5, math.sqrt(3)/2], device=device, dtype=dtype)
    C_std_2d_center = (A_std_2d + B_std_2d + C_std_2d) / 3

    centers_std_2d = torch.stack([
        (C_std_2d_center + A_std_2d) / 2,
        (C_std_2d_center + B_std_2d) / 2,
        (C_std_2d_center + C_std_2d) / 2
    ], dim=0)  # [3, 2]

    r = torch.norm(A_std_2d - C_std_2d_center) / 2

    # build local coordinate
    A_in = triangles[:, 0, :]  # [B, 3]
    B_in = triangles[:, 1, :]  # [B, 3]
    C_in = triangles[:, 2, :]  # [B, 3]

    Origin = A_in  # [B, 3]

    e1 = B_in - A_in  # [B, 3]
    norm_e1 = torch.norm(e1, dim=1, keepdim=True)  # [B, 1]
    u = e1 / norm_e1  # [B, 3]

    e2 = C_in - A_in  # [B, 3]

    # Compute the normal vector
    N = torch.cross(e1, e2, dim=1)  # [B, 3]
    norm_N = torch.norm(N, dim=1, keepdim=True)  # [B, 1]
    N = N / norm_N  # [B, 3]

    v = torch.cross(N, u, dim=1)  # [B, 3]

    # B,C local coordinate
    norm_e1_scalar = norm_e1[:, 0]  # [B]
    B_in_2d = torch.stack([norm_e1_scalar, torch.zeros(B, device=device, dtype=dtype)], dim=1)  # [B, 2]

    x_c = torch.sum(e2 * u, dim=1)  # [B]
    y_c = torch.sum(e2 * v, dim=1)  # [B]
    C_in_2d = torch.stack([x_c, y_c], dim=1)  # [B, 2]

    # 2D affine transform
    sqrt3_over_2 = math.sqrt(3) / 2
    x_b = B_in_2d[:, 0]  # [B]

    x_c = C_in_2d[:, 0]  # [B]
    y_c = C_in_2d[:, 1]  # [B]

    a = x_b  # [B]
    c = torch.zeros_like(a)  # [B]
    d = y_c / sqrt3_over_2  # [B]
    b = (x_c - a * 0.5) / sqrt3_over_2  # [B]

    T_2d = torch.zeros((B, 2, 2), device=device, dtype=dtype)  # [B, 2, 2]
    T_2d[:, 0, 0] = a
    T_2d[:, 0, 1] = b
    T_2d[:, 1, 0] = c
    T_2d[:, 1, 1] = d

    t_2d = torch.zeros((B, 2), device=device, dtype=dtype)  # [B, 2]

    # centers
    centers_std_2d_T = centers_std_2d.t()  # [2, 3]
    centers_in_2d = torch.matmul(T_2d, centers_std_2d_T.unsqueeze(0).expand(B, -1, -1))  # [B, 2, 3]
    centers_in_2d = centers_in_2d.transpose(1, 2)  # [B, 3, 2]
    centers_in_2d = centers_in_2d + t_2d.unsqueeze(1)  # [B, 3, 2]

    centers = Origin.unsqueeze(1) + centers_in_2d[:, :, 0].unsqueeze(2) * u.unsqueeze(1) + centers_in_2d[:, :, 1].unsqueeze(2) * v.unsqueeze(1)  # [B, 3, 3]

    # scales init as R / 2
    a = torch.norm(B_in - C_in, dim=1)  # [B]
    b = torch.norm(A_in - C_in, dim=1)  # [B]
    c = torch.norm(A_in - B_in, dim=1)  # [B]
    
    # Semi-perimeter
    s = (a + b + c) / 2  # [B]
    
    # Area using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))  # [B]
    
    # Circumcircle radius
    radii = (a * b * c) / (4.0 * area)  # [B]
    scales = radii.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, 2) / 2.0     # [B, 3, 2]

    # quats
    e1_2d = torch.tensor([1,0], device=device, dtype=dtype).expand(B, -1)  # [B, 2]
    e2_2d = torch.tensor([0,1], device=device, dtype=dtype).expand(B, -1)  # [B, 2]

    e1_3d = e1_2d[:, 0].unsqueeze(1) * u + e1_2d[:, 1].unsqueeze(1) * v  # [B, 3]
    e2_3d = e2_2d[:, 0].unsqueeze(1) * u + e2_2d[:, 1].unsqueeze(1) * v  # [B, 3]

    e1_3d = e1_3d / torch.norm(e1_3d, dim=1, keepdim=True)
    e2_3d = e2_3d / torch.norm(e2_3d, dim=1, keepdim=True)

    R_ellipse = torch.stack([e1_3d, e2_3d, N], dim=2)  # [B, 3, 3]

    # Correct for reflection
    det_R_ellipse = torch.det(R_ellipse)
    flip_mask = det_R_ellipse < 0
    if flip_mask.any():
        e2_3d[flip_mask] = -e2_3d[flip_mask]
        R_ellipse[flip_mask] = torch.stack([e1_3d[flip_mask], e2_3d[flip_mask], N[flip_mask]], dim=2)

    R_ellipse = R_ellipse[:, None].repeat(1, 3, 1, 1)    # [B, 6, 3, 3]

    centers = centers.to(torch.float32)
    scales = scales.to(torch.float32)
    R_ellipse = R_ellipse.to(torch.float32)

    return centers, scales, R_ellipse

def calculate_means(triangles):
    return torch.mean(triangles, dim=1)  # [B, 3]

def calculate_scales(triangles):
    A_in = triangles[:, 0, :]  # [B, 3]
    B_in = triangles[:, 1, :]  # [B, 3]
    C_in = triangles[:, 2, :]  # [B, 3]
    
    side_AB = B_in - A_in  # [B, 3]
    side_AC = C_in - A_in  # [B, 3]
    
    cross_prod = torch.cross(side_AB, side_AC, dim=1)  # [B, 3]
    area = 0.5 * torch.norm(cross_prod, dim=1)  # [B]
    
    a = torch.norm(B_in - C_in, dim=1)  # [B]
    b = torch.norm(A_in - C_in, dim=1)  # [B]
    c = torch.norm(A_in - B_in, dim=1)  # [B]
    perimeter = a + b + c  # [B]
    
    r = (2 * area) / perimeter  # [B]
    
    scales = r.unsqueeze(1).repeat(1, 2)  # [B, 2]
    
    return scales


def calculate_Rs(triangles):
    B = triangles.shape[0]

    device = triangles.device
    dtype = torch.float64
    triangles = triangles.to(dtype)

    # build local coordinate
    A_in = triangles[:, 0, :]  # [B, 3]
    B_in = triangles[:, 1, :]  # [B, 3]
    C_in = triangles[:, 2, :]  # [B, 3]

    e1 = B_in - A_in  # [B, 3]
    norm_e1 = torch.norm(e1, dim=1, keepdim=True)  # [B, 1]
    u = e1 / norm_e1  # [B, 3]

    e2 = C_in - A_in  # [B, 3]

    # Compute the normal vector
    N = torch.cross(e1, e2, dim=1)  # [B, 3]
    norm_N = torch.norm(N, dim=1, keepdim=True)  # [B, 1]
    N = N / norm_N  # [B, 3]

    v = torch.cross(N, u, dim=1)  # [B, 3]

    # quats
    e1_2d = torch.tensor([1,0], device=device, dtype=dtype).expand(B, -1)  # [B, 2]
    e2_2d = torch.tensor([0,1], device=device, dtype=dtype).expand(B, -1)  # [B, 2]

    e1_3d = e1_2d[:, 0].unsqueeze(1) * u + e1_2d[:, 1].unsqueeze(1) * v  # [B, 3]
    e2_3d = e2_2d[:, 0].unsqueeze(1) * u + e2_2d[:, 1].unsqueeze(1) * v  # [B, 3]

    e1_3d = e1_3d / torch.norm(e1_3d, dim=1, keepdim=True)
    e2_3d = e2_3d / torch.norm(e2_3d, dim=1, keepdim=True)

    R_ellipse = torch.stack([e1_3d, e2_3d, N], dim=2)  # [B, 3, 3]

    # Correct for reflection
    det_R_ellipse = torch.det(R_ellipse)
    flip_mask = det_R_ellipse < 0
    if flip_mask.any():
        e2_3d[flip_mask] = -e2_3d[flip_mask]
        R_ellipse[flip_mask] = torch.stack([e1_3d[flip_mask], e2_3d[flip_mask], N[flip_mask]], dim=2)

    R_ellipse = R_ellipse.to(torch.float32)

    return R_ellipse


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def init_colors_from_pc(init_points, init_colors, means, sh_degree=3, batch_size=128):
    """
    @Params:
    - init_points: [n, 3] (coordinates of the input points)
    - init_colors: [n, 3] (colors of the input points)
    - means: [N, 3] (centroids or target points to assign colors)
    
    @Returns:
    - rgbs: [N, 3] (colors assigned to the gs)
    """
    N = means.shape[0]
    n = init_points.shape[0]
    
    min_idxs = torch.zeros(N, dtype=torch.long)

    for i in range(0, N, batch_size):
        means_batch = means[i:i+batch_size]
        
        dists = torch.cdist(means_batch, init_points)  # [batch_size, n]
        
        _, min_idxs_batch = torch.min(dists, dim=1)  # [batch_size]
        
        min_idxs[i:i+batch_size] = min_idxs_batch

    rgbs = init_colors[min_idxs]  # [N, 3]

    return rgbs

def edge_equilateral_triangle_loss(vertices, faces):
    triangles = vertices[faces] # [F, 3, 3]

    edge_lengths = torch.cat(
        [
            torch.norm(triangles[:, 1] - triangles[:, 0], dim=1, keepdim=True),
            torch.norm(triangles[:, 2] - triangles[:, 1], dim=1, keepdim=True),
            torch.norm(triangles[:, 0] - triangles[:, 2], dim=1, keepdim=True)
        ],
        dim=1
    )   # [F, 3]

    loss = torch.mean(torch.abs(edge_lengths[:, 0] - edge_lengths[:, 1]) + 
                        torch.abs(edge_lengths[:, 0] - edge_lengths[:, 2]) + 
                        torch.abs(edge_lengths[:, 1] - edge_lengths[:, 2]))
    
    return loss


def compute_quats(normals, mean_c):
    """
    Compute quats 
    
    @Params:
    normals: [N,3]
    mean_c:  [N,3]

    @Returns:
    quats: [N,4]
    """
    normals = normals.copy()
    multipler = np.where(np.einsum('ij,ij->i', -normals, mean_c) > 0, 1, -1)
    adjusted_normals = normals * multipler[:, np.newaxis]

    norms = np.linalg.norm(adjusted_normals, axis=1, keepdims=True)
    b = adjusted_normals / norms  # [N,3]

    a = np.array([0, 0, 1]) 
    dot_prod = b[:, 2]

    mask_same = dot_prod > 0.999999
    mask_opposite = dot_prod < -0.999999
    mask_else = ~(mask_same | mask_opposite)

    quats = np.zeros((len(normals), 4))

    quats[mask_same, 0] = 1.0 

    axis = np.array([0, 1, 0])
    quats[mask_opposite, 0] = 0.0
    quats[mask_opposite, 1:] = axis  

    b_else = b[mask_else]
    rotation_axis = np.cross(a, b_else)
    rotation_axis_norm = np.linalg.norm(rotation_axis, axis=1, keepdims=True)
    rotation_axis_normalized = rotation_axis / rotation_axis_norm

    rotation_angle = np.arccos(dot_prod[mask_else])
    s = np.sin(rotation_angle / 2)

    quats[mask_else, 0] = np.cos(rotation_angle / 2)
    quats[mask_else, 1:] = rotation_axis_normalized * s[:, np.newaxis]

    quats = torch.tensor(quats, dtype=torch.float32)
    return quats

def init_gs_from_mesh(
        cfg, 
        mesh_params, 
        device,
        feature_dim):
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

    uv_sum = torch.logit(u + v)
    u_ratio = torch.logit(u / (u + v))

    scale_lambda =  torch.logit(0.5 * torch.ones((num_faces * cfg.num_mesh2gs, 2), dtype=torch.float, device=device))

    rot_2d = torch.Tensor([[1, 0]]).repeat(num_faces * cfg.num_mesh2gs, 1).to(device)

    # rgbs = torch.rand((num_faces * cfg.num_mesh2gs, 3)).to(device)
    # colors = torch.zeros((num_faces * cfg.num_mesh2gs, (cfg.sh_degree + 1) ** 2, 3)).to(device)  # [F, K, 3]
    # colors[:, 0, :] = rgb_to_sh(rgbs)

    # sh0 = colors[:, :1, :]
    # shN = colors[:, 1:, :]

    opacities = torch.logit(cfg.init_opa * torch.ones((num_faces * cfg.num_mesh2gs), dtype=torch.float, device=device))   # [F]

    params = {
        "uv_sum": uv_sum,
        "u_ratio": u_ratio,
        "scale_lambda": scale_lambda,
        "rot_2d": rot_2d,
        "opacities": opacities
    }

    if feature_dim is None:
        rgbs = torch.rand((num_faces * cfg.num_mesh2gs, 3)).to(device)
        colors = torch.zeros((num_faces * cfg.num_mesh2gs, (cfg.sh_degree + 1) ** 2, 3)).to(device)
        colors[:, 0, :] = rgb_to_sh(rgbs)

        sh0 = colors[:, :1, :]
        shN = colors[:, 1:, :]

        params.update({
            "sh0": sh0,
            "shN": shN,
        })
    else:
        rgbs = torch.rand((num_faces * cfg.num_mesh2gs, 3)).to(device)
        
        # features will be used for appearance and view-dependent shading
        features = torch.rand(num_faces * cfg.num_mesh2gs, feature_dim)  # [N, feature_dim]
        colors = torch.logit(rgbs)  # [N, 3]
        params.update({
            "features": features,
            "colors": colors,
        })

    params = {n: torch.nn.Parameter(v).to(device) for n, v in params.items()}

    ####################### Intermediate parameters #######################
    scales = torch.cat([scales * (torch.sigmoid(scale_lambda) * 1 + 0.5), torch.ones_like(scales)[:,:1]], dim=-1)
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

def calculate_ellipse_vertices(center, rotation, scaling):
    """
    Args:
        center: [P, 3]
        rotation: [P, 4] (quaternion [qx, qy, qz, qw])
        scaling: [P, 2]

    Returns:
        vertices: [P, 4, 3]
        # for P Gaussians, each Gaussian has 4 orthogonal elipsodial vertices, each vertex has 3 coordinates
    """
    P = center.shape[0]
    device = center.device

    # Create the unit circle points
    angles = torch.tensor([0, 90, 180, 270], dtype=torch.float32, device=device) * (torch.pi / 180)
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [4, 2]

    # Scale the unit circle points
    scaled_circle = scaling[:, None, :] * unit_circle[None, :, :]       # [P, 4, 2]

    # Padding to 3D space
    scaled_circle_3d = F.pad(scaled_circle, (0, 1), "constant", 0.0)    # [P, 4, 3]

    # converted to pytorch3d format
    rotation_converted = rotation[..., [3, 0, 1, 2]]                    # [P, 4], [qw, qx, qy, qz]

    # Rotation, quaternion_apply requires real part first
    rotated_circle = quaternion_apply(rotation_converted[:, None, :], scaled_circle_3d)  # [P, 4, 3]

    # Translate the points
    vertices = rotated_circle + center[:, None, :]  # [P, 4, 3]

    return vertices


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None] = None,
    normals: Union[torch.Tensor, None] = None,
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    weights=None,
    point_reduction: Union[str, None] = None,
    norm: int = 2,
    abs_cosine: bool = True,
):
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_normals
        cham_x = cham_x.max(1).values  # (N,)
    elif point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals


def chamfer_distance(x, y, x_normals=None, y_normals=None):
    x, x_lengths, x_normals = _handle_pointcloud_input(x, normals=x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, normals=y_normals)
    cham_x, cham_norm_x = _chamfer_distance_single_direction(x, y, x_lengths, y_lengths, x_normals, y_normals)
    return cham_x, cham_norm_x

def save_mesh(mesh_params, path):
    vertices = mesh_params["vertices"].detach().cpu().numpy()
    faces = mesh_params["faces"].cpu().numpy()
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh(path, mesh)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file,max_err=2,min_track_length=3):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)

    The colmap visualization filters out points with err greater than 2.0 and track length less than 3
    see: src/ui/model_viewer_widget.cc
        void ModelViewerWidget::UploadPointData(const bool selection_mode)
    """

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        count = 0
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            if error > max_err or track_length < min_track_length:
                continue

            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count +=1
    xyzs = np.delete(xyzs,np.arange(count,num_points),axis=0)
    rgbs = np.delete(rgbs, np.arange(count, num_points), axis=0)
    errors = np.delete(errors, np.arange(count, num_points), axis=0)
    return xyzs, rgbs, errors

def divide_point_cloud(points, mins=None, maxs=None, num_blocks=64):
    if mins is None:
        mins = points.min(axis=0) 
    if maxs is None:
        maxs = points.max(axis=0)
    blocks = []
    block_size = (maxs - mins) / num_blocks
    for i in tqdm(range(2 * num_blocks - 1)):
        for j in range(2 * num_blocks - 1):
            for k in range(2 * num_blocks - 1):
                # 定义每个 block 的范围
                block_min = mins + np.array([i, j, k]) / 2 * block_size
                block_max = block_min + block_size
                # 找到落在该块中的点
                block_points = points[
                    (points[:, 0] >= block_min[0]) & (points[:, 0] < block_max[0]) &
                    (points[:, 1] >= block_min[1]) & (points[:, 1] < block_max[1]) &
                    (points[:, 2] >= block_min[2]) & (points[:, 2] < block_max[2])
                ]
                if block_points.shape[0] > 10:  # 如果该块中有点
                    blocks.append(block_points)
    return blocks

def compute_convex_hulls_and_union(blocks):
    hulls = []
    for block in tqdm(blocks):
        hull = ConvexHull(block)
        hulls.append(hull)
    
    meshes = []
    for hull in tqdm(hulls):
        vertices = np.asarray(hull.points)
        faces = np.asarray(hull.simplices)
        meshes.append(trimesh.Trimesh(vertices, faces).convex_hull)
        
    return trimesh.boolean.union(meshes, check_volume=True)