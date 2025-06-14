import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager
from plyfile import PlyData, PlyElement
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
import open3d as o3d
import pickle

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    colors = (colors * 255).astype(np.uint8)
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    # DEBUG_TMP 
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    return positions, colors, normals

class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        dust3r_init: bool = False,
        test_every: int = 8,
        sky_mask_on: bool = False,
        dynamic_mask_on: bool = False,
        mono_depth_on: bool = False,
        load_plane_masks: bool = False,
        load_normal_maps: bool = False,
        instance_map_on: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.dust3r_init = dust3r_init
        self.test_every = test_every
        self.sky_mask_on = sky_mask_on
        self.dynamic_mask_on = dynamic_mask_on
        self.mono_depth_on = mono_depth_on
        self.load_plane_masks = load_plane_masks
        self.load_normal_maps = load_normal_maps
        
        self.instance_map_on = instance_map_on

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."


        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        
        # load paths of sky masks and dynamic masks
        mask_names = [f.replace('.jpg','.png').replace('.JPG', '.png')  for f in image_names]
        smask_dir = os.path.join(data_dir, "sky_masks")
        dmask_dir = os.path.join(data_dir, "dynamic_masks")
        smask_paths = [os.path.join(smask_dir, m) for m in mask_names] if sky_mask_on else None
        dmask_paths = [os.path.join(dmask_dir, m) for m in mask_names] if dynamic_mask_on else None

        # load paths of mono depth
        mono_depth_names = [f.replace('.jpg','.png').replace('.JPG', '.png')  for f in image_names]
        # mono_depth_names = [f.replace('.jpg','.npy').replace('.JPG', '.npy')  for f in image_names]
        mono_depth_dir = os.path.join(data_dir, "mono_depths")
        mono_depth_paths = [os.path.join(mono_depth_dir, d) for d in mono_depth_names] if mono_depth_on else None
        
        # Load Paths of plane masks
        plane_masks_names = [f.replace('.jpg','.npy').replace('.JPG', '.npy')  for f in image_names]
        plane_masks_dir = os.path.join(data_dir, "plane_mask_l")
        plane_masks_paths = [os.path.join(plane_masks_dir, d) for d in plane_masks_names] if load_plane_masks else None
        
        # Load Paths of normal maps
        normal_maps_names = [f.replace('.jpg','.png').replace('.JPG', '.png')  for f in image_names]
        normal_maps_dir = os.path.join(data_dir, "normal_maps")
        normal_maps_paths = [os.path.join(normal_maps_dir, d) for d in normal_maps_names] if load_normal_maps else None

        # load paths of instance maps
        imap_names = [f.replace('.jpg','.npy').replace('.png','.npy') for f in image_names]
        imap_dir = os.path.join(data_dir, "instance_maps")
        imap_paths = [os.path.join(imap_dir, i) for i in imap_names] if instance_map_on else None

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        points_normal = None
        point_indices = dict()

        if dust3r_init:
            print("Loading dust3r initializations...")
            dust3r_dir = os.path.join(data_dir, "dust3r_sparse/0/")
            positions, colors, normals = fetchPly(os.path.join(dust3r_dir,"points3D.ply"))
            # replace COLMAP points with dust3r points
            points = positions
            points_rgb = colors
            points_normal = normals

        # image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        # for point_id, data in manager.point3D_id_to_images.items():
        #     for image_id, _ in data:
        #         image_name = image_id_to_name[image_id]
        #         point_idx = manager.point3D_id_to_point3D_idx[point_id]
        #         point_indices.setdefault(image_name, []).append(point_idx)
        # point_indices = {
        #     k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        # }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.smask_paths = smask_paths
        self.dmask_paths = dmask_paths
        self.mono_depth_paths = mono_depth_paths
        self.plane_masks_paths = plane_masks_paths
        self.normal_maps_paths = normal_maps_paths

        self.imap_paths = imap_paths
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.points_normal = points_normal  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)

        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        smask = imageio.imread(self.parser.smask_paths[index])[..., :1] if self.parser.sky_mask_on else None
        dmask = imageio.imread(self.parser.dmask_paths[index])[..., :1] if self.parser.dynamic_mask_on else None
        imap = np.load(self.parser.imap_paths[index]) if self.parser.instance_map_on else None
        mono_depth = imageio.imread(self.parser.mono_depth_paths[index])[..., :1] if self.parser.mono_depth_on else None
        # mono_depth = np.load(self.parser.mono_depth_paths[index]) if self.parser.mono_depth_on else None
        plane_mask = np.load(self.parser.plane_masks_paths[index]) if self.parser.load_plane_masks else None

        # dmask = cv2.resize(dmask, image.shape[:2][::-1])[...,None]  #! HACK

        normal_map = imageio.imread(self.parser.normal_maps_paths[index])[..., :3] if self.parser.load_normal_maps else None
        
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        
        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            image = image[y : y + h, x : x + w]
            if smask is not None:
                smask = cv2.remap(smask, mapx, mapy, cv2.INTER_LINEAR)
                smask = smask[y : y + h, x : x + w]
            if dmask is not None:
                dmask = cv2.remap(dmask, mapx, mapy, cv2.INTER_LINEAR)
                dmask = dmask[y : y + h, x : x + w]
            if mono_depth is not None:
                mono_depth = cv2.remap(mono_depth, mapx, mapy, cv2.INTER_LINEAR)
                mono_depth = mono_depth[y : y + h, x : x + w]
            if plane_mask is not None:
                plane_mask = cv2.remap(plane_mask, mapx, mapy, cv2.INTER_LINEAR)
                plane_mask = plane_mask[y : y + h, x : x + w]
            if normal_map is not None:
                normal_map = cv2.remap(normal_map, mapx, mapy, cv2.INTER_LINEAR)
                normal_map = normal_map[y : y + h, x : x + w]
            if imap is not None:
                imap = cv2.remap(imap, mapx, mapy, cv2.INTER_LINEAR)
                imap = imap[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if smask is not None:
                smask = smask[y : y + self.patch_size, x : x + self.patch_size]
            if dmask is not None:
                dmask = dmask[y : y + self.patch_size, x : x + self.patch_size]
            if mono_depth is not None:
                mono_depth = mono_depth[y : y + self.patch_size, x : x + self.patch_size]
            if plane_mask is not None:
                plane_mask = plane_mask[y : y + self.patch_size, x : x + self.patch_size]
            if normal_map is not None:
                normal_map = normal_map[y : y + self.patch_size, x : x + self.patch_size]
            if imap is not None:
                imap = imap[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "image_name": self.parser.image_names[index].split('.')[0],
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if smask is not None:
            data["smask"] = torch.from_numpy(smask/255.0).float()
        if dmask is not None:
            data["dmask"] = torch.from_numpy(dmask/255.0).float()
        if mono_depth is not None:
            data["mono_depths"] = torch.from_numpy(mono_depth).float().unsqueeze(-1)    # [B, H, W, 1]
        if plane_mask is not None:
            data["plane_mask"] = torch.from_numpy(plane_mask)
        if normal_map is not None:
            data["normal_map"] = torch.from_numpy((normal_map / 255.0 - 0.5) * 2).float()
        if imap is not None:
            data["imap"] = torch.from_numpy(imap).int()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
