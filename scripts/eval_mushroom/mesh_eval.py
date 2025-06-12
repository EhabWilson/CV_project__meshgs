"""
adapted from go_surf scripts:
https://github.com/JingwenWang95/go-surf/blob/master/tools/mesh_metrics.py#L33
"""

import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Literal
from tqdm import tqdm

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
import tyro
import pyrender
from matplotlib import patches, pyplot as plt
from PIL import Image
from scipy.spatial import cKDTree

from examples.datasets.colmap import Parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def open3d_mesh_from_trimesh(tri_mesh):
    vertices = np.asarray(tri_mesh.vertices)
    faces = np.asarray(tri_mesh.faces)

    # Create open3d TriangleMesh object
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Assign vertices and faces to open3d mesh
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh


def trimesh_from_open3d_mesh(open3d_mesh):
    vertices = np.asarray(open3d_mesh.vertices)
    faces = np.asarray(open3d_mesh.triangles)

    # Create open3d TriangleMesh object
    tri_mesh = trimesh.Trimesh()
    tri_mesh.vertices = vertices
    tri_mesh.faces = faces
    return tri_mesh


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=5000)
    )
    transformation = reg_p2p.transformation
    return transformation


def render_depth_maps(mesh, poses, H, W, K, far=100.0, debug=False):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=far
    )
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY
    poses = deepcopy(poses)

    depth_maps = []
    for i, pose in enumerate(tqdm(poses, desc="Rendering depth maps", ncols=80)):
        pose[:3, 1:3] *= -1
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)

        if debug:
            global_max = np.max(depth)
            normalized_images = np.uint8((depth / global_max) * 255)
            colormapped_images = cv2.applyColorMap(
                normalized_images, cv2.COLORMAP_INFERNO
            )
            cv2.imwrite("depth_map_" + str(i) + ".png", colormapped_images)
        depth_maps.append(depth)

    return depth_maps


def cull_from_one_pose(
    points,
    pose,
    H,
    W,
    K,
    rendered_depth=None,
    depth_gt=None,
    remove_missing_depth=True,
    remove_occlusion=True,
):
    c2w = deepcopy(pose)
    # to OpenCV
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)
    eps = 0.02
    obs_mask = in_frustum
    # step 2: not occluded
    if remove_occlusion:
        obs_mask = in_frustum & (
            pz < (rendered_depth[v, u] + eps)
        )  # & (depth_gt[v, u] > 0.)

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum & (depth_gt[v, u] <= 0.0)
    else:
        invalid_mask = np.zeros_like(obs_mask)

    return obs_mask.astype(np.int32), invalid_mask.astype(np.int32)


def get_grid_culling_pattern(
    points,
    poses,
    H,
    W,
    K,
    rendered_depth_list=None,
    depth_gt_list=None,
    remove_missing_depth=True,
    remove_occlusion=True,
):

    obs_mask = np.zeros(points.shape[0])
    invalid_mask = np.zeros(points.shape[0])
    for i, pose in enumerate(tqdm(poses, desc="Getting grid culling pattern", ncols=80)):
        rendered_depth = (
            rendered_depth_list[i] if rendered_depth_list is not None else None
        )
        depth_gt = depth_gt_list[i] if depth_gt_list is not None else None
        obs, invalid = cull_from_one_pose(
            points,
            pose,
            H,
            W,
            K,
            rendered_depth=rendered_depth,
            depth_gt=depth_gt,
            remove_missing_depth=remove_missing_depth,
            remove_occlusion=remove_occlusion,
        )
        obs_mask = obs_mask + obs
        invalid_mask = invalid_mask + invalid

    return obs_mask, invalid_mask


# For meshes with backward-facing faces. For some reason the no_culling flag in pyrender doesn't work for depth maps
def render_depth_maps_doublesided(mesh, poses, H, W, K, far=100.0):
    K = torch.tensor(K).cuda().float()
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[
        :, [2, 1]
    ]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where(
            (depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map),
            depth_maps_2[i],
            depth_map,
        )
        depth_maps.append(depth_map)

    return depth_maps


def cut_projected_mesh(projection, predicted_mesh, type, kernel_size, dilate=True):
    # # Visualize
    # plt.figure(figsize=(10, 10))
    # ax = plt.gca()

    # # Invert y axis
    # ax.invert_yaxis()
    # plt.scatter(projection[:, 0], projection[:, 1], s=1)

    max_val = projection.max(axis=0)
    min_val = projection.min(axis=0)
    projection = ((projection - min_val) / (max_val - min_val) * 499).astype(np.int32)

    image = np.zeros((500, 500), dtype=np.uint8)

    for x, y in projection:
        image[y, x] = 255

    if kernel_size != None:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if dilate:
            rescale_image = cv2.dilate(image, kernel, iterations=1)
        elif dilate == False:
            rescale_image = cv2.erode(image, kernel, iterations=1)

        contours, _ = cv2.findContours(
            rescale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
    else:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert contour points back to their original scale
    contour_points = [
        np.array(c).squeeze() * (max_val - min_val) / 499 + min_val for c in contours
    ]


    # Filter the point cloud
    cloud_points = np.asarray(predicted_mesh.vertices)
    inside = np.zeros(len(cloud_points), dtype=bool)
    if type == "xy":
        project_points = cloud_points[:, :2]
    elif type == "xz":
        project_points = cloud_points[:, [0, 2]]
    elif type == "yz":
        project_points = cloud_points[:, 1:]

    inside = np.array(
        [
            any(
                patches.Path(contour).contains_point(point)
                for contour in contour_points
                if len(contour.shape) >= 2
            )
            for point in project_points
        ]
    )

    filtered_cloud = cloud_points[inside]

    old_to_new_indices = {old: new for new, old in enumerate(np.where(inside)[0])}

    triangles = np.asarray(predicted_mesh.triangles)
    for i in range(triangles.shape[0]):
        for j in range(3):
            if triangles[i, j] in old_to_new_indices:
                triangles[i, j] = old_to_new_indices[triangles[i, j]]
            else:
                triangles[i, j] = -1

    valid_triangles = (triangles != -1).all(axis=1)
    filtered_triangles = triangles[valid_triangles]

    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(filtered_cloud)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)

    return filtered_mesh


def cut_mesh(gt_mesh, pred_mesh, kernel_size, dilate=True):
    vertices = np.asarray(gt_mesh.vertices)
    # Extract vertex data and project it onto XY plane
    print("cutting xy plane")
    vertices_2d = vertices[:, :2]  # Keep only X and Y coordinates
    filtered_mesh = cut_projected_mesh(
        vertices_2d, pred_mesh, "xy", kernel_size, dilate=dilate
    )

    # Keep only X and Z coordinates
    print("cutting xz plane")
    vertices_2d = vertices[:, [0, 2]]
    filtered_mesh = cut_projected_mesh(
        vertices_2d, filtered_mesh, "xz", kernel_size, dilate=dilate
    )

    # Keep only Y and Z coordinates
    print("cutting yz plane")
    vertices_2d = vertices[:, 1:]
    filtered_mesh = cut_projected_mesh(
        vertices_2d, filtered_mesh, "yz", kernel_size, dilate=dilate
    )

    return filtered_mesh


def cull(poses, H, W, K, ref_mesh, ref_depths, mesh, depths):
    # mesh.remove_unreferenced_vertices()
    vertices = mesh.vertices
    triangles = mesh.faces
    obs_mask, invalid_mask = get_grid_culling_pattern(
        vertices,
        poses,
        H,
        W,
        K,
        rendered_depth_list=depths,
        depth_gt_list=ref_depths,
        remove_missing_depth=True,
        remove_occlusion=False,
    )
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    th1 = 3
    th2 = 0
    obs_mask = (obs1 > th1) | (obs2 > th1) | (obs3 > th1)
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = (inv1 > 0.7 * obs1) & (inv2 > 0.7 * obs2) & (inv3 > 0.7 * obs3)
    # invalid_mask = (inv1 > th2) & (inv2 > th2) & (inv3 > th2)

    valid_mask = obs_mask & (~invalid_mask)
    triangles_in_frustum = triangles[valid_mask, :]

    mesh = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh.remove_unreferenced_vertices()

    return mesh


def get_align_transformation(rec_meshfile, gt_meshfile):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    trans_init = np.eye(4)
    threshold = 0.2
    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1.000000e-08, relative_rmse=1.000000e-08, max_iteration=5000)
    )
    transformation = reg_p2p.transformation
    return transformation


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).astype(np.float32).mean() for t in thresholds]
    return in_threshold


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_colored_pcd(pcd, metric):
    cmap = plt.cm.get_cmap("jet")
    color = cmap(metric / 0.10)[..., :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def compute_metrics(mesh_pred, mesh_target):
    area_pred = int(mesh_pred.area * 1e4)
    area_tgt = int(mesh_target.area * 1e4)
    print("pred: {}, target: {}".format(area_pred, area_tgt))

    # iou, v_gt, v_pred = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(area_pred, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(area_tgt, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.array([0.05])

    # for every point in gt compute the min distance to points in pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    # color gt_point_cloud using completion
    # com_mesh = get_colored_pcd(pointcloud_tgt, completeness)

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    # color pred_point_cloud using completion
    # acc_mesh = get_colored_pcd(pointcloud_pred, accuracy)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    rst = {
        "Acc": accuracy,  # lower better
        "Comp": completeness,  # lower better
        "C-L1": chamferL1,  # lower better
        "NC": normals_correctness,  # higher better
        "F-score": F[0],  # higher better
    }

    return rst


def main(
    gt_mesh_path: Path,  # path to gt mesh folder
    pred_mesh_path: Path,  # path to the pred mesh ply
    output: Path,  # output path
    dataset_path: str,
    dataset: Literal["mushroom", "replica", "scannetpp"],
    load_depth: bool = False
):
    if not Path(output).exists():
        Path(output).mkdir(parents=True)

    gt_mesh = trimesh.load(str(gt_mesh_path), process=False)
    pred_mesh = trimesh.load(str(pred_mesh_path), process=False)

    # align gt mesh to colmap coordinate system
    if dataset == "mushroom":
        icp_transform_file = str(gt_mesh_path.parent / "icp_iphone.json") 
        with open(icp_transform_file, "r") as f:
            data = json.load(f)
            transform = np.array(data["gt_transformation"])
        transform = transform @ np.array(
            [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],]
        )
        gt_mesh = gt_mesh.apply_transform(np.linalg.inv(transform))
    elif dataset == "replica":
        # TODO: ICP align
        transformation = get_align_transformation(
            str(gt_mesh_path), str(pred_mesh_path)
        )
        gt_mesh = gt_mesh.apply_transform(transformation)
        np.save(str(output / "transformation.npy"), transformation)
        pred_mesh.export(str(output / "mesh.ply"))
        gt_mesh.export(str(output / "gt_mesh.ply"))
    elif dataset == "scannetpp":
        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset}.")

    # cull mesh based on depthmap
    parser = Parser(dataset_path)
    poses = parser.camtoworlds
    K = list(parser.Ks_dict.values())[0]
    W, H = list(parser.imsize_dict.values())[0]
    
    if load_depth:
        gt_depths_files = sorted(os.listdir(str(output / "gt_depths")))
        gt_depths = [np.load(os.path.join(str(output / "gt_depths"), f)) for f in gt_depths_files]
        pred_depths_files = sorted(os.listdir(str(output / "pred_depths")))
        pred_depths = [np.load(os.path.join(str(output / "pred_depths"), f)) for f in pred_depths_files]
    else:
        os.makedirs(str(output / "gt_depths"), exist_ok=True)
        os.makedirs(str(output / "gt_depths_vis"), exist_ok=True)
        os.makedirs(str(output / "pred_depths"), exist_ok=True)
        os.makedirs(str(output / "pred_depths_vis"), exist_ok=True)
        gt_depths = render_depth_maps_doublesided(gt_mesh, poses, H, W, K, far=10)
        pred_depths = render_depth_maps_doublesided(pred_mesh, poses, H, W, K, far=10)
        for i, depth in enumerate(gt_depths):
            np.save(str(output / "gt_depths" / f"{i:04d}.npy"), depth)
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            cv2.imwrite(str(output / "gt_depths_vis" / f"{i:04d}.png"), depth_normalized)
        for i, depth in enumerate(pred_depths):
            np.save(str(output / "pred_depths" / f"{i:04d}.npy"), depth)
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = depth_normalized.astype(np.uint8)
            cv2.imwrite(str(output / "pred_depths_vis" / f"{i:04d}.png"), depth_normalized)

    cull_pred_mesh = cull(poses, H, W, K, gt_mesh, gt_depths, pred_mesh, pred_depths)
    cull_gt_mesh = cull(poses, H, W, K, pred_mesh, pred_depths, gt_mesh, gt_depths)
    pred_mesh = cull_pred_mesh
    gt_mesh = cull_gt_mesh

    print(str(output / "mesh_cull.ply"))
    pred_mesh.export(str(output / "mesh_cull.ply"))
    gt_mesh.export(str(output / "gt_mesh_cull.ply"))

    # cut mesh based on the bounding box
    pred_mesh = open3d_mesh_from_trimesh(pred_mesh)
    gt_mesh = open3d_mesh_from_trimesh(gt_mesh)
    pred_mesh = cut_mesh(gt_mesh, pred_mesh, kernel_size=15, dilate=True)
    gt_mesh = trimesh_from_open3d_mesh(gt_mesh)
    pred_mesh = trimesh_from_open3d_mesh(pred_mesh)
    print(str(output / "mesh_cut.ply"))
    pred_mesh.export(str(output / "mesh_cut.ply"))
    gt_mesh.export(str(output / "gt_mesh_cut.ply"))

    rst = compute_metrics(pred_mesh, gt_mesh)
    print(f"Saving results to: {output / 'mesh_metrics.json'}")
    json.dump(rst, open(output / "mesh_metrics.json", "w"))

    print(rst)
    print(rst.values())


if __name__ == "__main__":
    tyro.cli(main)