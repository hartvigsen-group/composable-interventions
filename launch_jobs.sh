#!/bin/bash

# Define the common editor for all jobs, set this as required
common_editor="lora"

# Define the LM
lm_path="mistralai/Mistral-7B-Instruct-v0.3"

# Define sparsity levels and wbit levels to apply
sparsity_levels=(0.25 0.45 0.65)
wbit_levels=(2 4 8)

# Define different sets of configurations to be run
declare -a configs=(
    ### None ###
    "model_name=${lm_path} edit=none compression=none unlearn=none interventions=[] tag='None'"
    ## Edit only ###
    "model_name=${lm_path} edit=${common_editor} compression=none unlearn=none interventions=[edit] tag='${common_editor}_Edit'"
)

# Compress only - Wanda at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=none compression=wanda unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_Wanda${sparsity}%'")
done

# Compress only - SparseGPT at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=none compression=sparsegpt unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_SparseGPT${sparsity}%'")
done

# Edit then Compress - Wanda at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=wanda unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-to-Wanda${sparsity}%'")
done

# Edit then Compress - SparseGPT at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=sparsegpt unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${common_editor}-to-SparseGPT${sparsity}%'")
done

# Compress with Wanda then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=wanda unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='Wanda${sparsity}%-to-${common_editor}'")
done

# Compress with SparseGPT then Edit at different sparsity levels
for sparsity in "${sparsity_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=sparsegpt unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='SparseGPT${sparsity}%-to-${common_editor}'")
done

# Compress only - AWQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=none compression=awq unlearn=none interventions=[compress] wbits=${wbit} tag='Compress_AWQ${wbit}bit'")
done

# Compress only - GPTQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=none compression=gptq unlearn=none interventions=[compress] wbits=${wbit} tag='Compress_GPTQ${wbit}bit'")
done

# Edit then Compress - AWQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=awq unlearn=none interventions=[edit,compress] wbits=${wbit} tag='${common_editor}-to-AWQ${wbit}bit'")
done

# Edit then Compress - GPTQ at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=gptq unlearn=none interventions=[edit,compress] wbits=${wbit} tag='${common_editor}-to-GPTQ${wbit}bit'")
done

# Compress with AWQ then Edit at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=awq unlearn=none interventions=[compress,edit] wbits=${wbit} tag='AWQ${wbit}bit-to-${common_editor}'")
done

# Compress with GPTQ then Edit at different wbit levels
for wbit in "${wbit_levels[@]}"; do
    configs+=("model_name=${lm_path} edit=${common_editor} compression=gptq unlearn=none interventions=[compress,edit] wbits=${wbit} tag='GPTQ${wbit}bit-to-${common_editor}'")
done

# Loop through each configuration and launch a job
for cfg in "${configs[@]}"; do
    # if model_name=${lm_path} edit=none not in cfg
    if [[ $cfg != *"edit=none"* ]]; then
        sbatch run_exp.sh $cfg edit_dataset=zsre
        sbatch run_exp.sh $cfg edit_dataset=mquake
        sbatch run_exp.sh $cfg edit_dataset=counterfact
    else
        sbatch run_exp.sh $cfg
    fi
done
