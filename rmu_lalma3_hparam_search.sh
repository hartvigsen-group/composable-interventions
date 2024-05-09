#!/bin/bash

# "args": [
#                 // Overall
#                 "seed=42",
#                 "model=meta-llama/Meta-Llama-3-8B",
#                 "model_name=meta-llama/Meta-Llama-3-8B",
                
#                 // Editing
#                 "edit=False",
#                 "edit_dataset=zsre",

#                 // Compression
#                 "compress=False",
#                 "compress_first=False",
#                 "method=prune",
#                 "prune_method=wanda",
#                 "sparsity_ratio=0.25",
#                 "tag=Wanda_25%",

#                 // RMU Unlearning
#                 "+unlearn=True",
#                 "+unlearn_method=rmu",
#                 "+rmu_retain_corpora=[wikitext]",
#                 "+rmu_forget_corpora=[cyber-forget-corpus]",
#                 "+rmu_alpha=[10.0]",
#                 "+rmu_steering_coeff_list=[20,20]",
#                 "+rmu_lr=5e-05",
#                 "+rmu_min_len=0",
#                 "+rmu_max_len=2000",
#                 "+rmu_batch_size=8",
#                 // "+rmu_max_num_batches=500",
#                 "+rmu_max_num_batches=25",
#                 "+rmu_layer_id=4",
#                 "+rmu_layer_ids=[2,3,4]",
#                 "+rmu_param_ids=[3]",
#                 "+rmu_seed=42",
#             ]
#         }
#     ]
# }

model="meta-llama/Meta-Llama-3-8B"
layers=(31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3)
alphas=(1 10 100 1000 10000)
lrs=("1e-6" "5e-5" "1e-5" "1e-4")

number_combinations=$(( ${#layers[@]} * ${#alphas[~@]} * ${#lrs[@]} ))
echo "Number of combinations: $number_combinations"

for layer in "${layers[@]}"; do
    for alpha in "${alphas[@]}"; do
        for lr in "${lrs[@]}"; do
            layer_ids=[$(($layer-2)),$(($layer-1)),$layer]
            job_args="seed=42 wandb=online model=$model model_name=$model edit=False compress=False compress_first=False +unlearn=True +unlearn_method=rmu +rmu_retain_corpora=[wikitext] +rmu_forget_corpora=[cyber-forget-corpus] +rmu_alpha=[$alpha] +rmu_steering_coeff_list=[20,20] +rmu_lr=$lr +rmu_min_len=0 +rmu_max_len=2000 +rmu_batch_size=8 +rmu_max_num_batches=250 +rmu_layer_id=$layer +rmu_layer_ids=$layer_ids +rmu_param_ids=[$(($layer-1))] +rmu_seed=42"
            echo "Submitting job with args: $job_args"
            sbatch run_exp10x.sh $job_args
        done
    done
done