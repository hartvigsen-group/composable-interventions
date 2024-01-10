#!/bin/bash

#SBATCH --job-name=hydra-grid-search
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/hydra_job_%A_%a.out
#SBATCH --error=slurm_logs/hydra_job_%A_%a.err
#SBATCH --array=1-10

# Load necessary modules or activate virtual environment
source activate llm_310

# Run the Python script with Hydra's grid search
# The SLURM_ARRAY_TASK_ID environment variable will be different for each job in the array
python main.py --multirun edit_set=$SLURM_ARRAY_TASK_ID number_of_edits=50 edit=False compress=False save_ckpt=False method=prune