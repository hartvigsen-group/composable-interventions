#!/bin/bash

# Define the common editor for all jobs, set this as required
common_editor="ft"

# Define sparsity levels to apply
sparsity_levels=(0.25 0.45 0.65)

# Define different sets of configurations to be run
declare -a configs=(
    ### None ###
    "edit=none compression=none unlearn=none interventions=[] tag='None'"
    ## Edit only ###
    "edit=${common_editor} compression=none unlearn=none interventions=[edit] tag='${common_editor}_Edit'"
)

# Compress only - Wanda at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=none compression=wanda unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress\_Wanda${sparsity}\\%'")
done

# Compress only - SparseGPT at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=none compression=sparsegpt unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress\_SparseGPT${sparsity}\\%'")
done

# Edit then Compress - Wanda at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=wanda unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-rightarrow-Wanda${sparsity}\\%'")
done

# Edit then Compress - SparseGPT at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=sparsegpt unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-rightarrow-SparseGPT${sparsity}\\%'")
done

# Compress with Wanda then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=wanda unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='Wanda${sparsity}\\%-rightarrow-${common_editor}'")
done

# Compress with SparseGPT then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=sparsegpt unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='SparseGPT${sparsity}\\%-rightarrow-${common_editor}'")
done

# Loop through each configuration and launch a job
for cfg in "${configs[@]}"; do
    # Use sbatch for SLURM and passing Hydra config overrides
    sbatch run_exp10x.sh -- $cfg
done
