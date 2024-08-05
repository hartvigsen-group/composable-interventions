#!/bin/bash

#gd only ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=none edit_dataset=mquake compression=none unlearn=gd interventions=[unlearn] tag="gd-none"

# ##gd then Edit ###
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-memit"
# counterfact
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-memit"
# zsre
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=gd interventions=[unlearn,edit] tag="gd-memit"

# ##Edit then gd ###
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=gd interventions=[edit,unlearn] tag="lora-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=gd interventions=[edit,unlearn] tag="ft-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=gd interventions=[edit,unlearn] tag="memit-gd"
# counterfact
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=gd interventions=[edit,unlearn] tag="lora-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=gd interventions=[edit,unlearn] tag="ft-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=gd interventions=[edit,unlearn] tag="memit-gd"
# zsre
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=gd interventions=[edit,unlearn] tag="lora-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=gd interventions=[edit,unlearn] tag="ft-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=gd interventions=[edit,unlearn] tag="memit-gd"

##gd then Compress ###
# # GPTQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[unlearn,compress] wbits=2 tag="gd-gptq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[unlearn,compress] wbits=3 tag="gd-gptq3bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[unlearn,compress] wbits=4 tag="gd-gptq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[unlearn,compress] wbits=8 tag="gd-gptq8bit"
# # AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=2 tag="gd-awq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=3 tag="gd-awq3bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=4 tag="gd-awq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=5 tag="gd-awq5bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=6 tag="gd-awq6bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[unlearn,compress] wbits=8 tag="gd-awq8bit"
# # Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.25 tag="gd-wanda0.25\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.35 tag="gd-wanda0.35\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.45 tag="gd-wanda0.45\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.55 tag="gd-wanda0.55\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.65 tag="gd-wanda0.65\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.75 tag="gd-wanda0.75\%"
# # SparseGPT
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.25 tag="gd-sparsegpt0.25\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.35 tag="gd-sparsegpt0.35\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.45 tag="gd-sparsegpt0.45\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.55 tag="gd-sparsegpt0.55\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.65 tag="gd-sparsegpt0.65\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[unlearn,compress] sparsity_ratio=0.75 tag="gd-sparsegpt0.75\%"

# ##Compress then gd ###
# # GPTQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[compress,unlearn] wbits=2 tag="gptq2bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[compress,unlearn] wbits=3 tag="gptq3bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[compress,unlearn] wbits=4 tag="gptq4bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=gd interventions=[compress,unlearn] wbits=8 tag="gptq8bit-gd"
# # AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=2 tag="awq2bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=3 tag="awq3bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=4 tag="awq4bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=5 tag="awq5bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=6 tag="awq6bit-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=gd interventions=[compress,unlearn] wbits=8 tag="awq8bit-gd"
# # Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.25 tag="wanda0.25\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.35 tag="wanda0.35\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.45 tag="wanda0.45\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.55 tag="wanda0.55\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.65 tag="wanda0.65\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.75 tag="wanda0.75\%-gd"
# # SparseGPT
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.25 tag="sparsegpt0.25\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.35 tag="sparsegpt0.35\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.45 tag="sparsegpt0.45\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.55 tag="sparsegpt0.55\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.65 tag="sparsegpt0.65\%-gd"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=gd interventions=[compress,unlearn] sparsity_ratio=0.75 tag="sparsegpt0.75\%-gd"