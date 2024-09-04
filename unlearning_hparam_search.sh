#!/bin/bash

unlearning_type=$1
model_name=$2
model_tag=$3
min_num_freeze_layers=$4
max_num_freeze_layers=$5

learning_rates=(5e-6 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3)
num_training_samples=(2000 1000 500 450 400 350 300 250 200 150 100 50 25 10)
freeze_layers=($(seq $min_num_freeze_layers $max_num_freeze_layers))

# print the number of combinations
echo "Learning Rates: ${#learning_rates[@]}"
echo "Training Sample Sizes: ${#num_training_samples[@]}"
echo "Freeze Layers: ${#freeze_layers[@]}"


if [ $unlearning_type == "ga" ]; then
    num_runs=0
    for freeze_layer in "${freeze_layers[@]}"; do
        for lr in "${learning_rates[@]}"; do
            for train_sample_size in "${num_training_samples[@]}"; do
                num_runs=$((num_runs+1))
                echo run_exp.sh model_name=$model_name tag="ga_${model_tag}_hparam_search" unlearn=ga interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_train_size=$train_sample_size ga_lr=$lr
            done
        done
    done


    echo "Number of combinations: ${num_runs}"
fi