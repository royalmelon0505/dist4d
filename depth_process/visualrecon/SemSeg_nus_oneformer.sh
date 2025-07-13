source /data/miniconda3/bin/activate
conda activate mvs_py38
cd ./dist4d/depth_process/visualrecon
indice=$1
exp_root=./dist4d/depth_process/nus_Rdepth
exp_name=${indice}_scene

scene_root=${exp_root}/${exp_name}

python ./SemSeg/SemSeg_nus_oneformer.py --scene_root=$scene_root --gpu_num=$2 --mp_num=$3