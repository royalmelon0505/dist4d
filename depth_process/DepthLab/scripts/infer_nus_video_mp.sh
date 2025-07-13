#!/usr/bin/env bash
set -e
set -x

# export CUDA_VISIBLE_DEVICES=0

# cd ./dist4d/depth_process

indice=$1
exec 2>&1
exp_root="./depth_process/nus_Rdepth"
exp_name=${indice}_scene
python generate_Rdepth_condition.py --save_path $exp_root \
                                    --save_folder_name $exp_name \
                                    --index $indice \
                                    --mvs_input_root "./mvs_input_root"


cd ./DepthLab

bash scripts/infer_nus_onecam.sh 0 ${exp_root} ${exp_name} 0 &
bash scripts/infer_nus_onecam.sh 1 ${exp_root} ${exp_name} 1 &

bash scripts/infer_nus_onecam.sh 0 ${exp_root} ${exp_name} 2 &
bash scripts/infer_nus_onecam.sh 1 ${exp_root} ${exp_name} 3 &

bash scripts/infer_nus_onecam.sh 0 ${exp_root} ${exp_name} 4 &
bash scripts/infer_nus_onecam.sh 1 ${exp_root} ${exp_name} 5 &
wait

