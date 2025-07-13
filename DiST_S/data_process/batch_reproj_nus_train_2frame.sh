Start_idx=$1
End_idx=$2
source /data/miniconda3/bin/activate
conda activate pre_py39
cd ./data_process/

for (( indice=$Start_idx; indice<=$End_idx; indice+=2 )); 
do 
    indice_p=$((indice + 1))
    echo "Executing with indice = $indice $indice_p"
    python reproj_nus_scene_mp_2frame.py  --scene_idx=$indice --is_trainset&
    python reproj_nus_scene_mp_2frame.py  --scene_idx=$indice_p --is_trainset&
    wait
done