#!/usr/bin/env bash
set -e
set -x
export CUDA_VISIBLE_DEVICES=$1
exp_root=$2
exp_name=$3
cam_use=$4

checkpoint_root=./depth_process/DepthLab/checkpoints

pretrained_model_name_or_path="$checkpoint_root/marigold-depth-v1-0"
image_encoder_path="$checkpoint_root/CLIP-ViT-H-14-laion2B-s32B-b79K"
denoising_unet_path="$checkpoint_root/DepthLab/denoising_unet.pth"
reference_unet_path="$checkpoint_root/DepthLab/reference_unet.pth"
mapping_path="$checkpoint_root/DepthLab/mapping_layer.pth"

python infer_nus_video_mp.py  \
    --seed 1234 \
    --denoise_steps 5 \
    --processing_res 768 \
    --normalize_scale 1 \
    --strength 0.8 \
    --pretrained_model_name_or_path $pretrained_model_name_or_path --image_encoder_path $image_encoder_path \
    --denoising_unet_path $denoising_unet_path \
    --reference_unet_path $reference_unet_path \
    --mapping_path $mapping_path \
    --scene_root ${exp_root}/${exp_name}/\
    --output_dir ${exp_root}/${exp_name}/\
    --blend \
    --use_mvs_pcd \
    --cam_use ${cam_use}