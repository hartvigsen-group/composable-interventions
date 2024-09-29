#!/bin/bash

#SBATCH --job-name=hydra-grid-search
#SBATCH --account=hartvigsen_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=11:59:00
#SBATCH --output=slurm_logs/hydra_job_%A_%a.out
#SBATCH --error=slurm_logs/hydra_job_%A_%a.err
# ignore udc-an36-1
#SBATCH --exclude=udc-an36-1,udc-an36-13
# Load necessary modules or activate virtual environment
source activate lm-compose

python -m lm_compose wandb=online $@
