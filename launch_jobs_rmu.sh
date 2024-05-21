#!/bin/bash

##RMU only ###
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=none unlearn=rmu interventions=[unlearn] tag="rmu-none"

##RMU then Compress ###
# # GPTQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=2 tag="rmu-gptq2bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=4 tag="rmu-gptq4bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=8 tag="rmu-gptq8bit"
# # AWQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=2 tag="rmu-awq2bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=4 tag="rmu-awq4bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=8 tag="rmu-awq8bit"
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-wanda0.25\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-wanda0.45\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-wanda0.65\%"
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-sparsegpt0.25\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-sparsegpt0.45\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-sparsegpt0.65\%"

# ##Compress then RMU ###
# # GPTQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=2 tag="gptq2bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=4 tag="gptq4bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=8 tag="gptq8bit-rmu"
# # AWQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=2 tag="awq2bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=4 tag="awq4bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=8 tag="awq8bit-rmu"
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="wanda0.25\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="wanda0.45\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="wanda0.65\%-rmu"
# # SparseGPT
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="sparsegpt0.25\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="sparsegpt0.45\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="sparsegpt0.65\%-rmu"


# ##RMU then Edit ###
# sbatch run_exp.sh save_ckpt=False wandb=online edit=lora compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-lora"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-ft"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=memit compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-memit"

# ##Edit then RMU ###
# sbatch run_exp.sh save_ckpt=False wandb=online edit=lora compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft compression=none unlearn=rmu interventions=[edit,unlearn] tag="ft-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=memit compression=none unlearn=rmu interventions=[edit,unlearn] tag="memit-rmu"

##RMU then Compress ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=2 tag="rmu-gptq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=4 tag="rmu-gptq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=8 tag="rmu-gptq8bit"

##Compress then RMU ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=2 tag="gptq2bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=4 tag="gptq4bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=8 tag="gptq8bit-rmu"