# GaussianSplatting

## install
base docker: 2dgs_image
```
pip3 install -v -e .
cd examples && pip3 install -r requirements.txt 
```
## Data Preparation
Dust3r Init
- command
```python
cd InstantSplat
CUDA_VISIBLE_DEVICES=0 python coarse_init_infer.py --img_base_path /mnt/mnt_0/galaxea/operation_perception/indoor/2024_1022_kitchen_full --interval 30
```

Mono Depth from Depth-Anything-v2
- command 
```python
cd DepthAnythingV2

python distribute.py 
    --img-path /mnt/mnt_0/galaxea/operation_perception/indoor/2024_1022_kitchen_full/images \
    --outdir /mnt/mnt_0/galaxea/operation_perception/indoor/2024_1022_kitchen_full/mono_depths \
    --gpu 0 1 2 5 6 \
    --pred-only \ 
    --grayscale
```


## Training
To train a scene, simply use
```bash
python simple_trainer_2dgs.py --data_dir <path to COLMAP dataset> --result_dir <path to result>
```
To render and export mesh, run
```bash
python simple_trainer_2dgs.py --data_dir <path to COLMAP dataset> --result_dir <path to result> --ckpt <paths to all ckpts>
```
Commandline arguments for training:
```bash
# Arguments related to data loading
--data_dir          # Path to COLMAP dataset
--data_factor       # Downsample factor for the dataset
--result_dir        # Path to result
--sky_mask_on       # Load sky masks
--dynamic_mask_on   # Load dynamic masks
--global_scale      # A global scaler that applies to the scene size
--near_plane        # Near plane clipping distance
--far_plane         # Far plane clipping distance
--packed            # Use packed mode for rasterization, this leads to less memory usage but slightly slower.

# Arguments related to training settings
--max_steps         # Number of training steps
--batch_size        # Batch size. Learning rates are scaled automatically
--test_every          # Every N images there is a test image
--steps_scaler      # A global factor to scale the number of training steps
--eval_steps        # Steps to evaluate the model
--save_steps        # Steps to save the model
--disable_viewer    # Disable viewer
--port              # Port for the viewer server

# Arguments related to testing
--ckpt              # Paths to the .pt file. If provide, it will skip training and render a video

# Arguments related to optimization
--focal_loss_on     # focal l1 loss
--ssim_lambda       # Weight for SSIM loss. (l1_lambda = 1-ssim_lambda)
--depth_loss        # Enable depth loss
--depth_lambda      # Weight for depth loss
--normal_loss       # Enable normal consistency loss
--normal_lambda_2dgs    # Weight for 2dgs normal loss
--normal_lambda_dnsplatter  # Weight for dnsplatter normal loss
--normal_start_iter # Iteration to start normal consistency regulerization
--dist_loss         # Enable distortion loss
--dist_lambda       # Weight for distortion loss
--dist_start_iter   # Iteration to start distortion loss regulerization
--sky_lambda        # Weight for sky loss

# Arguments related to densification
--prune_opa         # GSs with opacity below this value will be pruned
--grow_grad2d       # GSs with image plane gradient above this value will be split/duplicated
--grow_scale3d      # GSs with scale below this value will be duplicated. Above will be split
--prune_scale3d     # GSs with scale above this value will be pruned.
--refine_start_iter # Start refining GSs after this iteration
--refine_stop_iter  # Stop refining GSs after this iteration
--reset_every       # Reset opacities every this steps
--refine_every      # Refine GSs every this steps
--pause_refine_after_reset   # Pause refining GSs until this number of steps after reset
--absgrad           # Use absolute gradient for pruning. This typically requires larger grow_grad2d

# Arguments related to log
--wandb_on          # Log with wandb
--wandb_project     # Wandb project name
--wandb_group       # Wandb group name
--log_every         # How often to dump information to wandb
--log_save_image    # Save training images to wandb

# Arguments related to pose refinement
--pose_opt          # Enable camera optimization.
--pose_opt_lr       # Learning rate for camera optimization
--pose_opt_reg      # Regularization for camera optimization as weight decay
--pose_noise        # Add noise to camera extrinsics. This is only to test the camera pose optimization.

# Arguments related to appearance embedding
--app_opt           # Enable appearance optimization. (experimental)
--app_embed_dim     # Appearance embedding dimension
--app_opt_lr        # Learning rate for appearance optimization
--app_opt_reg       # Regularization for appearance optimization as weight decay
```


## Extract mesh
```
CUDA_VISIBLE_DEVICES=0 python simple_trainer_2dgs.py --data_dir <> --result_dir  <>  --gs_ply <> --disable_viewer 
```