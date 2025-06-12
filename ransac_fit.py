import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.pipeline import make_pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_plane(normals, mask):
    """
    
    Args:
        normals (numpy.ndarray): [H, W, 3] 的 normal map.
        mask (numpy.ndarray): [H, W] 的 mask.

    Returns:
        bool: True if is plane; else False
    """
    masked_normals = normals[mask > 0]

    if masked_normals.size == 0:
        return False

    std = np.std(masked_normals, axis=0)

    return np.all(std < 0.04)

def process_file(mask_file):
    base_name = os.path.splitext(mask_file)[0]
    mask_path = os.path.join(mask_folder, mask_file)
    output_path = os.path.join(output_folder, mask_file.replace('.pkl', '.npy'))

    # Check if the output file already exists
    if os.path.exists(output_path):
        return

    # Read sam masks
    with open(mask_path, 'rb') as f:
        data = pickle.load(f)
    masks = data['masks']  # list of [H*W] arrays
    
    # Read normal and rgb
    normal_path = os.path.join(normal_folder, base_name + '.png')
    image_path = os.path.join(image_folder, base_name + '.jpg')
    
    normal_map = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)

    if normal_map is None:
        print(f"Error reading {normal_path}")
        return
    
    normal_map = normal_map.astype(np.float32) / 255.0  # normalization
    H, W, _ = normal_map.shape

    filtered_masks = []
    
    for i, mask in enumerate(masks):
        mask = mask.reshape(H, W)
        
        normal_values = normal_map[mask.astype(bool)]
        
        if len(normal_values) == 0:
            continue
        
        # RANSAC
        X = normal_values[:, :2]
        y = normal_values[:, 2]
        
        ransac = RANSACRegressor(LinearRegression())          
        ransac.fit(X, y)
        
        inlier_mask = ransac.inlier_mask_
        
        # SKIP 
        if np.sum(inlier_mask) / len(inlier_mask) < 0.5:
            continue
        
        filtered_mask = np.zeros_like(mask)
        filtered_mask[mask.astype(bool)] = inlier_mask
        
        # Filter by normal variance
        if is_plane(normal_map, filtered_mask):
            filtered_masks.append(filtered_mask)
    
    # Save the filtered masks as a numpy array, even if empty
    if filtered_masks:
        # Calculate areas for each mask
        areas = [np.sum(mask) for mask in filtered_masks]
        # Sort masks by area, largest first
        sorted_indices = np.argsort(areas)
        
        # Create a final mask of size [H, W] and fill with largest masks first
        compressed_mask = np.zeros((H, W), dtype=np.uint8)
        for i in sorted_indices:
            compressed_mask[filtered_masks[i] > 0] = i + 1  # Assign index as plane label
    
        # Save the compressed mask
        np.save(output_path, compressed_mask)
        print(f"Results saved to {output_path}")
    else:
        # If no valid mask, save an empty array
        np.save(output_path, np.zeros((H, W), dtype=np.uint8))
        print(f"Empty result saved to {output_path}")

    # Optionally visualize every 10th image
    if (int(base_name[-4:])) % 10 == 0:
        image = cv2.imread(image_path)
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        for mask in filtered_masks:
            color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
            image[mask.astype(bool)] = (
                0.5 * image[mask.astype(bool)] + 0.5 * color
            ).astype(np.uint8)
        
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f"{base_name}.png"))

# paths
mask_folder = '/mnt/mnt_0/galaxea/operation_perception/indoor/kitchen/sam_masks_s'
normal_folder = '/mnt/mnt_0/galaxea/operation_perception/indoor/kitchen/normal_maps'
image_folder = '/mnt/mnt_0/galaxea/operation_perception/indoor/kitchen/images'
output_folder = '/mnt/mnt_0/galaxea/operation_perception/indoor/kitchen/plane_masks_s'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Use ThreadPoolExecutor to process files concurrently
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(process_file, mask_file) for mask_file in sorted(os.listdir(mask_folder))]
    
    for future in as_completed(futures):
        try:
            future.result()  # Get the result (or exception if any)
        except Exception as exc:
            print(f"Generated an exception: {exc}")
