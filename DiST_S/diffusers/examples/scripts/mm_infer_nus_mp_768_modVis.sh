# only infer +1m 
python examples/inference_mm_mp_modVis.py \
    --model_path ../ckpt/mm_multi_cam_hybridD_valid_mask_768_mod_cycle_p/checkpoint-20000 \
    --img_root ../reproj_nus/val  \
    --output_dir ../output_vis \
    --pesudo_flag_name frame_movecam2egoy+1 \
    --with_valid_mask \
    --video_width=768 \
    --video_height=448 \

