Start_idx=$1
End_idx=$2
source /data/miniconda3/bin/activate
conda activate pre_py39
cd ./data_process/

for (( indice=$Start_idx; indice<=$End_idx; indice+=1 )); 
do 
    # indice_p=$((indice + 1))
    # echo "Executing with indice = $indice $indice_p"
    echo "Executing with indice = $indice "
    python reproj_nus_scene_movecam.py  --scene_idx=$indice --mod_y_d=1
    python reproj_nus_scene_movecam.py  --scene_idx=$indice --mod_y_d=2
    python reproj_nus_scene_movecam.py  --scene_idx=$indice --mod_y_d=4
    wait
done