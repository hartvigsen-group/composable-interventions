#!/bin/bash

# Define the common editor for all jobs, set this as required
common_editor="memit"

# Define sparsity levels and wbit levels to apply
sparsity_levels=(0.25 0.45 0.65)
wbit_levels=(2 4 8)

# Define different sets of configurations to be run
declare -a configs=(
    ### None ###
    # "edit=none compression=none unlearn=none interventions=[] tag='None'"
    ## Edit only ###
    "edit=${common_editor} compression=none unlearn=none interventions=[edit] tag='${common_editor}_Edit'"
)

# # Compress only - Wanda at different sparsity levels
# for sparsity in "${sparsity_levels[@]}"; do
#     configs+=("edit=none compression=wanda unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_Wanda${sparsity}%'")
# done

# # Compress only - SparseGPT at different sparsity levels
# for sparsity in "${sparsity_levels[@]}"; do
#     configs+=("edit=none compression=sparsegpt unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_SparseGPT${sparsity}%'")
# done

# Edit then Compress - Wanda at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=wanda unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-to-Wanda${sparsity}%'")
done

# Edit then Compress - SparseGPT at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=sparsegpt unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-to-SparseGPT${sparsity}%'")
done

# Compress with Wanda then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=wanda unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='Wanda${sparsity}%-to-${common_editor}'")
done

# Compress with SparseGPT then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("edit=${common_editor} compression=sparsegpt unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='SparseGPT${sparsity}%-to-${common_editor}'")
done

# Compress only - AWQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=none compression=awq unlearn=none interventions=[compress] wbit=${wbit} tag='Compress_AWQ${wbit}bit'")
done

# Compress only - GPTQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=none compression=gptq unlearn=none interventions=[compress] wbit=${wbit} tag='Compress_GPTQ${wbit}bit'")
done

# Edit then Compress - AWQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=${common_editor} compression=awq unlearn=none interventions=[edit,compress] wbit=${wbit} tag='${common_editor}-to-AWQ${wbit}bit'")
done

# Edit then Compress - GPTQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=${common_editor} compression=gptq unlearn=none interventions=[edit,compress] wbit=${wbit} tag='${common_editor}-to-GPTQ${wbit}bit'")
done

# Compress with AWQ then Edit at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=${common_editor} compression=awq unlearn=none interventions=[compress,edit] wbit=${wbit} tag='AWQ${wbit}bit-to-${common_editor}'")
done

# Compress with GPTQ then Edit at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("edit=${common_editor} compression=gptq unlearn=none interventions=[compress,edit] wbit=${wbit} tag='GPTQ${wbit}bit-to-${common_editor}'")
done

# Loop through each configuration and launch a job
for cfg in "${configs[@]}"; do
    # Use sbatch for SLURM and passing Hydra config overrides
    # sbatch run_exp10x.sh $cfg
    echo sbatch run_exp10x.sh $cfg
done
