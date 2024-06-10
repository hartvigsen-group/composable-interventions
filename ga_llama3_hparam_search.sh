#!/bin/bash

ga_train_sizes=(2000 1000 500 450 400 350 300 250 200 150 100 50 25 10)
learning_rates=(5e-6 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3)
jobs_args=()


# for learnign rate
for lr in "${learning_rates[@]}"; do
    # Default ga_train_size
    jobs_args+=("tag=ga_llama3_hparam_search unlearn=ga interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_lr=$lr")

    # Have runs with different number of training samples
    for train_sample_size in "${ga_train_sizes[@]}"; do
        job_args="tag=ga_llama3_hparam_search unlearn=ga interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_train_size=$train_sample_size ga_lr=$lr"
        jobs_args+=("$job_args")
    done
done

for job_args in "${jobs_args[@]}"; do
    echo "Submitting job with args: $job_args"
    sbatch run_exp.sh $job_args
done

echo "Number of combinations: ${#jobs_args[@]}"