#!/bin/bash

#SBATCH --job-name=hydra-grid-search
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:59:00
#SBATCH --output=slurm_logs/hydra_job_%A_%a.out
#SBATCH --error=slurm_logs/hydra_job_%A_%a.err
#SBATCH --array=1-10

# Load necessary modules or activate virtual environment
source activate lm-compose

# Run the Python script with Hydra's grid search
# The SLURM_ARRAY_TASK_ID environment variable will be different for each job in the array
python -m lm_compose edit_set=$SLURM_ARRAY_TASK_ID seed=$SLURM_ARRAY_TASK_ID edit_dataset=counterfact wandb=online $@
