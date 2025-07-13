python examples/inference_mm_mp_cycle_proj.py \
    --model_path ../ckpt/mm_multi_cam_hybridD_valid_mask_768/checkpoint-12500 \
    --save_root ../reproj_nus/cycle_reproj/ \
    --with_valid_mask \
    --video_width=768 \
    --video_height=448 \
    --is_trainset \