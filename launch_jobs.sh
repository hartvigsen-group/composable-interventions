#!/bin/bash

# Define different sets of parameters
params=(
    # "edit=False compress=False tag=v2_none"
    "edit=True compress=False tag=v2_memit"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.25 tag=v2_wanda25"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.35 tag=v2_wanda35"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.45 tag=v2_wanda45"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.55 tag=v2_wanda55"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.65 tag=v2_wanda65"
    "edit=False compress=True method=prune prune_method=wanda sparsity_ratio=0.75 tag=v2_wanda75"
    "edit=False compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.25 tag=v2_sparsegpt25"
    "edit=False compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.50 tag=v2_sparsegpt50"
    "edit=False compress=True method=prune prune_method=sparsegpt sparsity_ratio=0.75 tag=v2_sparsegpt75"
    "edit=True compress=True method=prune prune_method=wanda sparsity_ratio=0.35 tag=v2_memit_wanda25"
    "edit=True compress=True method=prune prune_method=wanda sparsity_ratio=0.55 tag=v2_memit_wanda55"
    "edit=True compress=True method=prune prune_method=wanda sparsity_ratio=0.75 tag=v2_memit_wanda75"
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    sbatch run_exp10x.sh $p
done
