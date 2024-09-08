#!/bin/bash

model=$1
save_path=$2
params=(
    ### None ###
    # "compress=False sparsity_ratio=0.0 tag=none"
    ## Compress only ###
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.25 tag=RMU-rightarrow-SparseGPT$_\{25%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.35 tag=RMU-rightarrow-SparseGPT$_\{35%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.45 tag=RMU-rightarrow-SparseGPT$_\{45%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.55 tag=RMU-rightarrow-SparseGPT$_\{55%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.65 tag=RMU-rightarrow-SparseGPT$_\{65%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.75 tag=RMU-rightarrow-SparseGPT$_\{75%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=sparsegpt sparsity_ratio=0.85 tag=RMU-rightarrow-SparseGPT$_\{85%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.25 tag=RMU-rightarrow-Wanda$_\{25\%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.35 tag=RMU-rightarrow-Wanda$_\{35%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.45 tag=RMU-rightarrow-Wanda$_\{45%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.55 tag=RMU-rightarrow-Wanda$_\{55%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.65 tag=RMU-rightarrow-Wanda$_\{65%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.75 tag=RMU-rightarrow-Wanda$_\{75%\}$ wandb=online"
    "interventions=[unlearn,compress] prune_method=wanda sparsity_ratio=0.85 tag=RMU-rightarrow-Wanda$_\{85%\}$ wandb=online"
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    sleep 5
    echo "Beginning: $p"
    sbatch run_exp.sh $p
done
