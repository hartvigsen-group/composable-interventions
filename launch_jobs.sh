#!/bin/bash

# Define different sets of parameters
params=(
    "edit=False compress=True method=prune sparsity_ratio=0.25 tag=exp_sparsegpt25"
    "edit=False compress=True method=prune sparsity_ratio=0.50 tag=exp_sparsegpt50"
    "edit=False compress=True method=prune sparsity_ratio=0.75 tag=exp_sparsegpt75"
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    sbatch run_exp10x.sh $p
done
