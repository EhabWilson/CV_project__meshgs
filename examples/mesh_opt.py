import numpy as np
from scipy.spatial import Delaunay

def post_process_delaunay(vertices, delaunay_triangles, max_edge_length=None):
    tri_vertices = vertices[delaunay_triangles]  # [T, 3, 3]
    edge1 = tri_vertices[:, 1] - tri_vertices[:, 0]  # [T, 3]
    edge2 = tri_vertices[:, 2] - tri_vertices[:, 1]  # [T, 3]
    edge3 = tri_vertices[:, 0] - tri_vertices[:, 2]  # [T, 3]

    lengths1 = np.linalg.norm(edge1, axis=1)  # [T]
    lengths2 = np.linalg.norm(edge2, axis=1)  # [T]
    lengths3 = np.linalg.norm(edge3, axis=1)  # [T]

    all_lengths = np.concatenate([lengths1, lengths2, lengths3])  # [3*T]

    if max_edge_length is None:
        median_length = np.median(all_lengths)
        q75, q25 = np.percentile(all_lengths, [75, 25])
        iqr = q75 - q25
        threshold = median_length + 5 * iqr
    else:
        threshold = max_edge_length

    valid_triangles_mask = (lengths1 < threshold) & (lengths2 < threshold) & (lengths3 < threshold)

    filtered_delaunay_triangles = delaunay_triangles[valid_triangles_mask]

    return filtered_delaunay_triangles

def mesh_merge(vertices, faces, sel_index, normal_map, sam_mask):
    """
    @Params:
    - vertices  : [N,3]
    - faces     : [F,3]
    - sel_index : [N]
    - normal_map: [H,W,3], in world coordinate system
    - sam_mask  : [H,W]

    @Returns:
    - new_faces: [F',3]
    """
    masked_to_original_idx = np.arange(len(vertices))[sel_index]

    sel_vertices = vertices[sel_index]                  # [M, 3]
    n = normal_map[sam_mask].mean(axis=0)               
    n = n / np.linalg.norm(n)

    p0 = np.mean(sel_vertices, axis=0)

    # Project the selected points onto the plane
    vectors_to_plane = sel_vertices - p0  # [M, 3]
    distances_along_normal = np.dot(vectors_to_plane, n)  # [M]
    x_proj = sel_vertices - np.outer(distances_along_normal, n)  # [M, 3]

    # Define a 2D coordinate system in the plane
    ref_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if np.abs(np.dot(n, ref_vector)) > 0.99:
        ref_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    u = np.cross(n, ref_vector)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    # Compute 2D coordinates in the plane
    x_proj_vectors = x_proj - p0  # [M, 3]
    x_proj_2D_u = np.dot(x_proj_vectors, u)  # [M]
    x_proj_2D_v = np.dot(x_proj_vectors, v)  # [M]
    sel_points2D = np.column_stack((x_proj_2D_u, x_proj_2D_v))  # [M, 2]

    # Perform Delaunay triangulation
    tri = Delaunay(sel_points2D)
    delaunay_triangles = tri.simplices  # [T, 3]

    delaunay_triangles = masked_to_original_idx[delaunay_triangles]
    
    new_faces = post_process_delaunay(vertices, delaunay_triangles)

    faces_mask = sel_index[faces]
    faces_mask_sum = faces_mask.sum(axis=1)
    faces_to_keep = faces_mask_sum < 3

    return faces_to_keep, new_faces