Start_idx=$1
End_idx=$2
gpu_num=$3
mp_num=$4
for (( indice=$Start_idx; indice<=$End_idx; indice++ )); 
do 
    echo "Executing with indice = $indice, gpu $gpu_num, mp_num $mp_num"
    bash SemSeg_nus_oneformer.sh $indice $gpu_num $mp_num
done
