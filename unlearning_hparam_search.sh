#!/bin/bash

unlearning_type=$1
model_name=$2
model_tag=$3
min_num_freeze_layers=$4
max_num_freeze_layers=$5

# learning_rates=(5e-6 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3)
learning_rates=(5e-5)
num_training_samples=(2000 1000 500 450 400 350 300 250 200 150 100 50 25 10)
freeze_layers=($(seq $min_num_freeze_layers $max_num_freeze_layers))
num_runs=0

if [ $unlearning_type == "ga" ]; then
    for freeze_layer in "${freeze_layers[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for train_sample_size in "${num_training_samples[@]}"; do
                num_runs=$((num_runs + 1))
                sbatch run_exp.sh model_name=$model_name tag="ga_${model_tag}_hparam_search" unlearn=ga interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_train_size=$train_sample_size ga_lr=$lr
            done
        done
    done
fi

retain_weights=(1 2 4 6 8 10 20 40 60 80 100)
if [ $unlearning_type == "gd" ]; then
    num_runs=0
    for lr in "${learning_rates[@]}"; do
        for train_sample_size in "${num_training_samples[@]}"; do
            for retain_weight in "${retain_weights[@]}"; do
                num_runs=$((num_runs + 1))
                sbatch run_exp.sh model_name=$model_name tag="gd_${model_tag}_hparam_search" unlearn=gd interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_train_size=$train_sample_size ga_lr=$lr ga_retain_weight=$retain_weight
            done
        done
    done
fi

alphas=(10000 1000 100 10 1)
layers=(17 3)
num_batches=(300 250 200 150 100)
if [ $unlearning_type == "rmu" ]; then
    for num_batch in "${num_batches[@]}"; do
        for layer in "${layers[@]}"; do
            for alpha in "${alphas[@]}"; do
                num_runs=$((num_runs + 1))
                sbatch run_exp.sh model_name=$model_name tag="rmu_${model_tag}_hparam_search" unlearn=rmu interventions=[unlearn] seed=42 wandb=online save_ckpt=False rmu_alpha=[$alpha,$alpha] rmu_layer_id=$layer rmu_max_num_batches=$num_batch
            done
        done
    done
fi

sbatch "Number of combinations: ${num_runs}"