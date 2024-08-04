#!/bin/bash
MODEL="mistralai/Mistral-7B-Instruct-v0.3"

# none
sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=none compression=none unlearn=none interventions=[] tag="none"

#rmu only ###
sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=none compression=none unlearn=rmu interventions=[unlearn] tag="rmu-none" rmu_alpha=[10000,10000] rmu_max_num_batches=150 rmu_layer_id=6 rmu_layer_ids=[4,5,6]

# lora only
sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=lora compression=none unlearn=none interventions=[edit] tag="lora-none"

# wanda only
sparsity_levels=(0.25 0.45 0.65)
for sparsity in "${sparsity_levels[@]}"; do
    sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=none compression=wanda unlearn=none interventions=[compress] sparsity_ratio=${sparsity} tag="Compress_Wanda${sparsity}%"
done

# rmu then lora
sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=lora compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-lora" rmu_alpha=[10000,10000] rmu_max_num_batches=150 rmu_layer_id=6 rmu_layer_ids=[4,5,6]

# lora then rmu
sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=lora compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rmu" rmu_alpha=[10000,10000] rmu_max_num_batches=150 rmu_layer_id=6 rmu_layer_ids=[4,5,6]

# rmu then wanda
for sparsity in "${sparsity_levels[@]}"; do
    sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=${sparsity} tag="rmu-wanda${sparsity}%" rmu_alpha=[10000,10000] rmu_max_num_batches=150 rmu_layer_id=6 rmu_layer_ids=[4,5,6]
done

# wanda then rmu
for sparsity in "${sparsity_levels[@]}"; do
    sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=${sparsity} tag="wanda-rmu${sparsity}%" rmu_alpha=[10000,10000] rmu_max_num_batches=150 rmu_layer_id=6 rmu_layer_ids=[4,5,6]
done

# lora then wanda
for sparsity in "${sparsity_levels[@]}"; do
    sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=lora compression=wanda unlearn=none interventions=[edit,compress] sparsity_ratio=${sparsity} tag="lora-wanda${sparsity}%"
done

# wanda then lora
for sparsity in "${sparsity_levels[@]}"; do
    sbatch run_exp.sh save_ckpt=False wandb=online model_name=$MODEL edit=lora compression=wanda unlearn=none interventions=[compress,edit] sparsity_ratio=${sparsity} tag="wanda-lora${sparsity}%"
done