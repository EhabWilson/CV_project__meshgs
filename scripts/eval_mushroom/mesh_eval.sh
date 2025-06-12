REPLICA_SCENE_DIR="/mnt/mnt_1/yuhang.cao/data/replica_GeoGaussian"
REPLICA_RESULT_DIR="/mnt/mnt_1/yuhang.cao/exp_2dgs"
REPLICA_SCENE_LIST="office0 office1 office2 office3 office4"

for SCENE in $REPLICA_SCENE_LIST;
do
    echo "Running $SCENE"
    python mesh_eval.py \
        --gt_mesh_path /home/yuhang.cao/data/replica_mesh/${SCENE}_mesh_triangle_align.ply \
        --pred_mesh_path ${REPLICA_RESULT_DIR}/${SCENE}/fuse_post.ply \
        --output ${REPLICA_RESULT_DIR}/${SCENE}/metric \
        --dataset_path ${REPLICA_SCENE_DIR}/${SCENE} \
        --dataset replica
done


# MUSHROOM_SCENE_DIR="/mnt/mnt_0/galaxea/operation_perception/OpenDataset/MushRoom"
# MUSHROOM_RESULT_DIR="/mnt/mnt_0/galaxea/yuhang.cao/experiments/results_2dgs/mushroom"
# MUSHROOM_SCENE_LIST="computer kokko vr_room honka sauna coffee_room"

# for SCENE in $MUSHROOM_SCENE_LIST;
# do
#     echo "Running $SCENE"
#     python mesh_eval.py \
#         --gt_mesh_path /home/yuhang.cao/data/Mushroom_mesh/${SCENE}/gt_mesh.ply \
#         --pred_mesh_path ${MUSHROOM_RESULT_DIR}/${SCENE}/fuse_post.ply \
#         --output ${MUSHROOM_RESULT_DIR}/${SCENE}/metric \
#         --dataset_path ${MUSHROOM_SCENE_DIR}/${SCENE}
# done


# SCANNETPP_SCENE_DIR="/mnt/mnt_1/yuhang.cao/data/Scannetpp"
# SCANNETPP_RESULT_DIR="/mnt/mnt_1/yuhang.cao/exp_2dgs"
# SCANNETPP_SCENE_LIST="8b5caf3398 b20a261fdf"

# for SCENE in $SCANNETPP_SCENE_LIST;
# do
#     echo "Running $SCENE"
#     python mesh_eval.py \
#         --gt_mesh_path ${SCANNETPP_SCENE_DIR}/${SCENE}/scans/mesh_aligned_0.05.ply \
#         --pred_mesh_path ${SCANNETPP_RESULT_DIR}/${SCENE}/fuse_post.ply \
#         --output ${SCANNETPP_RESULT_DIR}/${SCENE}/metric \
#         --dataset_path ${SCANNETPP_SCENE_DIR}/${SCENE} \
#         --dataset scannetpp
# done