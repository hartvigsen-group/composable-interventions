#!/bin/bash

# Define different sets of parameters
params=(
    # Edit then RMU
    "interventions=[edit,unlearn] tag=Test-FT-RMU edit=ft wandb=online"
    "interventions=[edit,unlearn] tag=Test-LoRA-RMU edit=lora wandb=online"
    "interventions=[edit,unlearn] tag=Test-MEMIT-RMU edit=memit wandb=online"

    # RMU then Edit
    "interventions=[unlearn,edit] tag=Test-RMU-FT edit=ft wandb=online"
    "interventions=[unlearn,edit] tag=Test-RMU-LoRA edit=lora wandb=online"
    "interventions=[unlearn,edit] tag=Test-RMU-MEMIT edit=memit wandb=online"

    # Edit only
    "interventions=[edit] tag=Test-FT edit=ft wandb=online"
    "interventions=[edit] tag=Test-LoRA edit=lora wandb=online"
    "interventions=[edit] tag=Test-MEMIT edit=memit wandb=online"

    # RMU only
    "interventions=[unlearn] tag=Test-RMU wandb=online"

    # None (Evals Only)
    "interventions=[] tag=Test-None wandb=online"
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    echo "Submitting job with parameters: $p"
    sbatch run_exp.sh $p
done
