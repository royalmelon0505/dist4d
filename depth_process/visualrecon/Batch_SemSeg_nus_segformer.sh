Start_idx=$1
End_idx=$2

for (( indice=$Start_idx; indice<=$End_idx; indice++ )); 
do 
    echo "Executing with indice = $indice"
    bash SemSeg_nus_segformer.sh $indice
done
