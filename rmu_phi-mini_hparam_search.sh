#!/bin/bash

model="microsoft/Phi-3-mini-4k-instruct"
alphas=(10000 1000 100 10 1)
layers=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)
num_batches=(1000 500 450 400 350 300 250 200 150 100)

for num_batch in "${num_batches[@]}"; do
    for layer in "${layers[@]}"; do
        for alpha in "${alphas[@]}"; do
            layer_ids=[$(($layer-2)),$(($layer-1)),$layer]
            job_args="tag=rmu_mistral_hparam_search model_name=${model} unlearn=rmu interventions=[unlearn] seed=42 wandb=online save_ckpt=False rmu_alpha=[$alpha,$alpha] rmu_max_num_batches=$num_batch rmu_layer_id=$layer rmu_layer_ids=$layer_ids"
            # echo "Submitting job with args: $job_args"
            echo sbatch run_exp.sh $job_args
        done
    done
done

number_combinations=$(( ${#num_batches[@]} * ${#layers[@]} * ${#alphas[@]} * ${#lrs[@]} ))
echo "Number of combinations: $number_combinations"