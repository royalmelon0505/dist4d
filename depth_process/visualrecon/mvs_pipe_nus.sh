INDICE=$1

conda activate pre_py39
cd ./dist4d/depth_process
python generate_MVS_data.py  --index_list $INDICE

cd ./dist4d/depth_process/visualrecon
conda deactivate
conda activate mvs_py38
bash run.sh configs.only_mvs_nus_sem $INDICE

conda deactivate
conda activate pre_py39

bash run.sh configs.only_mvs_nus $INDICE