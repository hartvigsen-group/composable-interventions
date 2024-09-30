#!/bin/bash

# ##RMU only ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=none edit_dataset=mquake compression=none unlearn=rmu interventions=[unlearn] tag="rmu-none"

# ##RMU then Compress ###
# GPTQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=2 tag="rmu-gptq2bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=3 tag="rmu-gptq3bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=4 tag="rmu-gptq4bit"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[unlearn,compress] wbits=8 tag="rmu-gptq8bit"
# AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=2 tag="rmu-awq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=3 tag="rmu-awq3bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=4 tag="rmu-awq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=5 tag="rmu-awq5bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=6 tag="rmu-awq6bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[unlearn,compress] wbits=8 tag="rmu-awq8bit"
# Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-wanda0.25\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.35 tag="rmu-wanda0.35\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-wanda0.45\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.55 tag="rmu-wanda0.55\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-wanda0.65\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.75 tag="rmu-wanda0.75\%"
# # # SparseGPT
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-sparsegpt0.25\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.35 tag="rmu-sparsegpt0.35\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-sparsegpt0.45\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.55 tag="rmu-sparsegpt0.55\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-sparsegpt0.65\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.75 tag="rmu-sparsegpt0.75\%"

# # ##Compress then RMU ###
# # # GPTQ
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=2 tag="gptq2bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=3 tag="gptq3bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=4 tag="gptq4bit-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=rmu interventions=[compress,unlearn] wbits=8 tag="gptq8bit-rmu"
# # # AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=2 tag="awq2bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=3 tag="awq3bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=4 tag="awq4bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=5 tag="awq5bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=6 tag="awq6bit-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=rmu interventions=[compress,unlearn] wbits=8 tag="awq8bit-rmu"
# # # Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="wanda0.25\%-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.35 tag="wanda0.35\%-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="wanda0.45\%-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.55 tag="wanda0.55\%-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="wanda0.65\%-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.75 tag="wanda0.75\%-rmu"
# # # SparseGPT
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="sparsegpt0.25\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.35 tag="sparsegpt0.35\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="sparsegpt0.45\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.55 tag="sparsegpt0.55\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="sparsegpt0.65\%-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.75 tag="sparsegpt0.75\%-rmu"

# ##RMU then Edit ###
# # mquake
# sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-lora"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-ft"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-memit"
# # counterfact
# sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-lora"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-ft"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-memit"
# zsre
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-lora"
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-ft"
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-memit"

# ##Edit then RMU ###
# # mquake
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rmu"
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=rmu interventions=[edit,unlearn] tag="ft-rmu"
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=rmu interventions=[edit,unlearn] tag="memit-rmu"
# # counterfact
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rmu"
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=rmu interventions=[edit,unlearn] tag="ft-rmu"
# sbatch run_exp10x.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=rmu interventions=[edit,unlearn] tag="memit-rmu"
# zsre
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rmu"
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=rmu interventions=[edit,unlearn] tag="ft-rmu"
sbatch run_exp10x.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=rmu interventions=[edit,unlearn] tag="memit-rmu"
