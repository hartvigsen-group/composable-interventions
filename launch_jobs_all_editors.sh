#!/bin/bash

# Define the common editor for all jobs, set this as required
editors=("ft")

# Define sparsity levels and wbit levels to apply
sparsity_levels=(0.25)
wbit_levels=(4)
# (0.25 0.35 0.45 0.55 0.65 0.75) and (2 3 4 5 6 8)

# Define different sets of configurations to be run
declare -a configs=(
    ### None ###
    "edit=none compression=none unlearn=none interventions=[] tag='None'"
)

for editor in "${editors[@]}"; do
   ## Edit only ###
    configs+=("edit=${editor} compression=none unlearn=none interventions=[edit] tag='${editor}_Edit'")
done

# # Compress only - Wanda at different sparsity levels
# for sparsity in "${sparsity_levels[@]}"; do
#     configs+=("edit=none compression=wanda unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_Wanda${sparsity}%'")
# done

# # Compress only - SparseGPT at different sparsity levels
# for sparsity in "${sparsity_levels[@]}"; do
#     configs+=("edit=none compression=sparsegpt unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag='Compress_SparseGPT${sparsity}%'")
# done

for editor in "${editors[@]}"; do
    # Edit then Compress - Wanda at different sparsity levels
    for sparsity in "${sparsity_levels[@]}"; do
        configs+=("edit=${editor} compression=wanda unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${editor}-to-Wanda${sparsity}%'")
    done
done

for editor in "${editors[@]}"; do
    # Edit then Compress - SparseGPT at different sparsity levels
    for sparsity in "${sparsity_levels[@]}"; do
        configs+=("edit=${editor} compression=sparsegpt unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag='${editor}-to-SparseGPT${sparsity}%'")
    done
done

for editor in "${editors[@]}"; do
    # Compress with Wanda then Edit at different sparsity levels
    for sparsity in "${sparsity_levels[@]}"; do
        configs+=("edit=${editor} compression=wanda unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='Wanda${sparsity}%-to-${editor}'")
    done
done

for editor in "${editors[@]}"; do
    # Compress with SparseGPT then Edit at different sparsity levels
    for sparsity in "${sparsity_levels[@]}"; do
        configs+=("edit=${editor} compression=sparsegpt unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag='SparseGPT${sparsity}%-to-${editor}'")
    done
done

# # Compress only - AWQ at different wbit levels
# for wbit in "${wbit_levels[@]}"; do
#     configs+=("edit=none compression=awq unlearn=none interventions=[compress] wbits=${wbit} tag='Compress_AWQ${wbit}bit'")
# done

# # Compress only - GPTQ at different wbit levels
# for wbit in "${wbit_levels[@]}"; do
#     configs+=("edit=none compression=gptq unlearn=none interventions=[compress] wbits=${wbit} tag='Compress_GPTQ${wbit}bit'")
# done

for editor in "${editors[@]}"; do
    # Edit then Compress - AWQ at different wbit levels
    for wbit in "${wbit_levels[@]}"; do
        configs+=("edit=${editor} compression=awq unlearn=none interventions=[edit,compress] wbits=${wbit} tag='${editor}-to-AWQ${wbit}bit'")
    done
done

for editor in "${editors[@]}"; do
    # Edit then Compress - GPTQ at different wbit levels
    for wbit in "${wbit_levels[@]}"; do
        configs+=("edit=${editor} compression=gptq unlearn=none interventions=[edit,compress] wbits=${wbit} tag='${editor}-to-GPTQ${wbit}bit'")
    done
done

for editor in "${editors[@]}"; do
    # Compress with AWQ then Edit at different wbit levels
    for wbit in "${wbit_levels[@]}"; do
        configs+=("edit=${editor} compression=awq unlearn=none interventions=[compress,edit] wbits=${wbit} tag='AWQ${wbit}bit-to-${editor}'")
    done
done

for editor in "${editors[@]}"; do
    # Compress with GPTQ then Edit at different wbit levels
    for wbit in "${wbit_levels[@]}"; do
        configs+=("edit=${editor} compression=gptq unlearn=none interventions=[compress,edit] wbits=${wbit} tag='GPTQ${wbit}bit-to-${editor}'")
    done
done

# Loop through each configuration and launch a job
for cfg in "${configs[@]}"; do
    # Use sbatch for SLURM and passing Hydra config overrides
    sbatch run_exp10x.sh $cfg
    echo sbatch run_exp10x.sh $cfg
done