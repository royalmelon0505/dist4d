export NCCL_P2P_LEVEL=NVL 
export MODEL_NAME="../pretrained/stable-video-diffusion-img2vid-xt" # SVD official
export DATASET_NAME='../reproj_nus/'
export HF_HOME="./huggingface"

work_dir=../ckpt
WORK=mm_multi_cam_hybridD_valid_mask_768
EXP=mm_multi_cam_hybridD_valid_mask_768


accelerate launch --config_file fsdp.yaml --main_process_port 12012 examples/train_svd_nus_mm.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --use_ema \
  --train_batch_size 1 \
  --video_width 768 \
  --video_height 448 \
  --gradient_accumulation_steps 4 \
  --enable_xformers_memory_efficient_attention \
  --max_train_steps 30000 \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" \
  --output_dir="$work_dir/$WORK" \
  --dataloader_num_workers 6 \
  --nframes 6 \
  --conditioning_dropout_prob=0.2 \
  --mismatch_aug_ratio=0.5 \
  --seed_for_gen=42 \
  --ddim \
  --checkpointing_steps 2500 \
  --tracker_project_name $EXP \
  --load_from_pkl \
  --checkpoints_total_limit 20 \
  --initialize_pseudoimg_encoder \
  --with_valid_mask 
  # --single_cam 


