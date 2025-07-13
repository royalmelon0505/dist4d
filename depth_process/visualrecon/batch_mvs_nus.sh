#!/bin/bash

for indice in {0..99}
do 
    echo "Executing with indice = $indice"
    bash mvs_pipe_nus.sh $indice
done