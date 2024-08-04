#!/bin/bash

# Constants
# layers=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
model="mistralai/Mistral-7B-Instruct-v0.3"

# Full search 5/11
# alphas=(1 10 100 1000 10000)
# layers=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
# num_batches=(100 150 200 250 300 350 400 450 500 1000)

# Reverse search 5/12
alphas=(10000 1000 100 10 1)
layers=(2 4 6 8 10 12 14 16 18 20 22 24 26 28 30)
num_batches=(1000 500 450 400 350 300 250 200 150 100)

for num_batch in "${num_batches[@]}"; do
    for layer in "${layers[@]}"; do
        for alpha in "${alphas[@]}"; do
            layer_ids=[$(($layer-2)),$(($layer-1)),$layer]
            # old args
            # job_args="tag=mustral_hparam_search seed=42 wandb=online save_ckpt=False model_name=$model edit=False compress=False compress_first=False +unlearn=True +unlearn_method=rmu +rmu_retain_corpora=[wikitext,wikitext] +rmu_forget_corpora=[bio-forget-corpus,cyber-forget-corpus] +rmu_alpha=[$alpha,$alpha] +rmu_steering_coeffs=[20,20] +rmu_lr=$lr +rmu_min_len=0 +rmu_max_len=2000 +rmu_batch_size=4 +rmu_max_num_batches=$num_batch +rmu_layer_id=$layer +rmu_layer_ids=$layer_ids +rmu_param_ids=[$(($layer))] +rmu_seed=42"
            
            # example args
            # jobs_args+=("tag=gd_mistral_hparam_search model_name=mistralai/Mistral-7B-Instruct-v0.3 unlearn=gd interventions=[unlearn] seed=42 wandb=online save_ckpt=False ga_train_size=$train_sample_size ga_lr=$lr ga_retain_weight=$retain_weight")

            # new args
            job_args="tag=rmu_mistral_hparam_search model_name=mistralai/Mistral-7B-Instruct-v0.3 unlearn=rmu interventions=[unlearn] seed=42 wandb=online save_ckpt=False rmu_alpha=[$alpha,$alpha] rmu_max_num_batches=$num_batch rmu_layer_id=$layer rmu_layer_ids=$layer_ids"
            
            echo "Submitting job with args: $job_args"
            sbatch run_exp.sh $job_args
        done
    done
done

number_combinations=$(( ${#num_batches[@]} * ${#layers[@]} * ${#alphas[@]} * ${#lrs[@]} ))
echo "Number of combinations: $number_combinations"