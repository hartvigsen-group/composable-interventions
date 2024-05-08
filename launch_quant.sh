#!/bin/bash

model=$1
save_path=$2
wbits=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
for b in "${wbits[@]}"; do
    echo "compress=True method=quant quant_method=autogptq wbits=$b tag=AutoGPTQ_wbit$b model=$model model_name=$model save_ckpt_path=$save_path"
    sbatch run_exp.sh "compress=True method=quant quant_method=autogptq wbits=$b tag=AutoGPTQ_wbit$b model=$model model_name=$model save_ckpt_path=$save_path"
    sleep 5
done