#!/bin/bash

# Experiment with three interventions with Llama. Permuted order of interventions.
# Unlearn: RMU
# Edit: LoRA
# Compresison: AWQ

wbit_levels=(2 4 8)
intervention_orders=("[unlearn,edit,compress]" "[unlearn,compress,edit]" "[edit,unlearn,compress]" "[edit,compress,unlearn]" "[compress,unlearn,edit]" "[compress,edit,unlearn]")
declare -a configs=()
declare -a category_name_map=( 
    ["[unlearn,edit,compress]"]="RMU-LoRA-AWQ"
    ["[unlearn,compress,edit]"]="RMU-AWQ-LoRA"
    ["[edit,unlearn,compress]"]="LoRA-RMU-AWQ"
    ["[edit,compress,unlearn]"]="LoRA-AWQ-RMU"
    ["[compress,unlearn,edit]"]="AWQ-RMU-LoRA"
    ["[compress,edit,unlearn]"]="AWQ-LoRA-RMU"
)

for order in "${intervention_orders[@]}"; do
    for wbit in "${wbit_levels[@]}"; do
        # have tag as the order of interventions but with th eintervention name. unlearn=rmu, edit=lora, compress=awq
        echo "${category_name_map[${order}]}-${wbit}bit"

        # configs+=("edit=${category_name_map[edit]} compression=${category_name_map[compress]} unlearn=${category_name_map[unlearn]} interventions=${order} wbits=${wbit} tag='Compress_AWQ${wbit}bit'")
        # configs+=("edit=lora compression=awq unlearn=rmu interventions=${order} wbits=${wbit} tag='${category_name_map[${order}]}-${wbit}bit'")
    done
done

# Loop through each configuration and launch a job
for cfg in "${configs[@]}"; do
    # Use sbatch for SLURM and passing Hydra config overrides
    echo sbatch run_exp.sh $cfg
    # echo sbatch run_exp10x.sh $cfg
done

# [RMU, LoRA, AWQ]

# [RMU, AWQ, LoRA]

# [LoRA, RMU, AWQ]

# [LoRA, AWQ, RMU]

# [AWQ, RMU, LoRA]

# [AWQ, LoRA, RMU]

