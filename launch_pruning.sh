#!/bin/bash

model=$1
save_path=$2
params=(
    ### None ###
    # "compress=False sparsity_ratio=0.0 tag=none"
    ## Compress only ###
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.25 tag=SparseGPT$_\{25%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.35 tag=SparseGPT$_\{35%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.45 tag=SparseGPT$_\{45%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.55 tag=SparseGPT$_\{55%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.65 tag=SparseGPT$_\{65%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.75 tag=SparseGPT$_\{75%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.85 tag=SparseGPT$_\{85%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.25 tag=Wanda$_\{25\%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.35 tag=Wanda$_\{35%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.45 tag=Wanda$_\{45%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.55 tag=Wanda$_\{55%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.65 tag=Wanda$_\{65%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.75 tag=Wanda$_\{75%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
    "compress=True method=prune prune_method=wanda sparsity_ratio=0.85 tag=Wanda$_\{85%\}$ model=$model model_name=$model save_ckpt_path=$save_path"
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    sleep 5
    echo "Beginning: $p" 
    sbatch run_exp.sh $p
done