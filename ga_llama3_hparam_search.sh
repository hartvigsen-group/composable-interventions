# #!/bin/bash

# # Constants
# # layers=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)
# model="meta-llama/Meta-Llama-3-8B"
# lrs=("5e-5")

# # Full search 5/11
# # alphas=(1 10 100 1000 10000)
# # layers=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)
# # num_batches=(100 150 200 250 300 350 400 450 500 1000)

# # Reverse search 5/12
# alphas=(10000 1000 100 10 1)
# layers=(17 16 15 14 13 12 11 10 9 8 7 6 5 4 3)
# num_batches=(1000 500 450 400 350 300 250 200 150 100)

# for num_batch in "${num_batches[@]}"; do
#     for layer in "${layers[@]}"; do
#         for alpha in "${alphas[@]}"; do
#             for lr in "${lrs[@]}"; do
#                 layer_ids=[$(($layer-2)),$(($layer-1)),$layer]
#                 job_args="tag=rmu_hparam_search seed=42 wandb=online save_ckpt=False edit=False compress=False compress_first=False +unlearn=True +unlearn_method=rmu +rmu_retain_corpora=[wikitext,wikitext] +rmu_forget_corpora=[bio-forget-corpus,cyber-forget-corpus] +rmu_alpha=[$alpha,$alpha] +rmu_steering_coeffs=[20,20] +rmu_lr=$lr +rmu_min_len=0 +rmu_max_len=2000 +rmu_batch_size=4 +rmu_max_num_batches=$num_batch +rmu_layer_id=$layer +rmu_layer_ids=$layer_ids +rmu_param_ids=[$(($layer))] +rmu_unlearn=ga seed=42"
#                 echo "Submitting job with args: $job_args"
#                 sbatch run_expx.sh $job_args
#             done
#         done
#     done
# done

# number_combinations=$(( ${#num_batches[@]} * ${#layers[@]} * ${#alphas[@]} * ${#lrs[@]} ))
# echo "Number of combinations: $number_combinations"

#!/bin/bash

ga_train_sample_sizes=(2000 1000 500 450 400 350 300 250 200 150 100 50 25 10)

jobs_args=()

# Default ga_train_sample_size
jobs_args+=("tag=ga_llama3_hparam_search unlearn=ga seed=42 wandb=online save_ckpt=False")

for train_sample_size in "${ga_train_sample_sizes[@]}"; do
    job_args="tag=ga_llama3_hparam_search unlearn=ga seed=42 wandb=online save_ckpt=False ga_train_sample_size=$train_sample_size"
    jobs_args+=("$job_args")
done

for job_args in "${jobs_args[@]}"; do
    echo "Submitting job with args: $job_args"
    sbatch run_exp.sh $job_args
done

echo "Number of combinations: ${#jobs_args[@]}"