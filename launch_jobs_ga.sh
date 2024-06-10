#!/bin/bash

#ga only ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=none edit_dataset=mquake compression=none unlearn=ga interventions=[unlearn] tag="ga-none"

# ##ga then Edit ###
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-memit"
# counterfact
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-memit"
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-lora"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=ga interventions=[unlearn,edit] tag="ga-memit"

# ##Edit then ga ###
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=mquake compression=none unlearn=ga interventions=[edit,unlearn] tag="lora-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=mquake compression=none unlearn=ga interventions=[edit,unlearn] tag="ft-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=mquake compression=none unlearn=ga interventions=[edit,unlearn] tag="memit-ga"
# counterfact
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=counterfact compression=none unlearn=ga interventions=[edit,unlearn] tag="lora-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=counterfact compression=none unlearn=ga interventions=[edit,unlearn] tag="ft-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=counterfact compression=none unlearn=ga interventions=[edit,unlearn] tag="memit-ga"
# mquake
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora edit_dataset=zsre compression=none unlearn=ga interventions=[edit,unlearn] tag="lora-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=ft edit_dataset=zsre compression=none unlearn=ga interventions=[edit,unlearn] tag="ft-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit edit_dataset=zsre compression=none unlearn=ga interventions=[edit,unlearn] tag="memit-ga"

##ga then Compress ###
# # GPTQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[unlearn,compress] wbits=2 tag="ga-gptq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[unlearn,compress] wbits=3 tag="ga-gptq3bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[unlearn,compress] wbits=4 tag="ga-gptq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[unlearn,compress] wbits=8 tag="ga-gptq8bit"
# # AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=2 tag="ga-awq2bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=3 tag="ga-awq3bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=4 tag="ga-awq4bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=5 tag="ga-awq5bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=6 tag="ga-awq6bit"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[unlearn,compress] wbits=8 tag="ga-awq8bit"
# # Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.25 tag="ga-wanda0.25\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.35 tag="ga-wanda0.35\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.45 tag="ga-wanda0.45\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.55 tag="ga-wanda0.55\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.65 tag="ga-wanda0.65\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.75 tag="ga-wanda0.75\%"
# # SparseGPT
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.25 tag="ga-sparsegpt0.25\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.35 tag="ga-sparsegpt0.35\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.45 tag="ga-sparsegpt0.45\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.55 tag="ga-sparsegpt0.55\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.65 tag="ga-sparsegpt0.65\%"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[unlearn,compress] sparsity_ratio=0.75 tag="ga-sparsegpt0.75\%"

# ##Compress then ga ###
# # GPTQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[compress,unlearn] wbits=2 tag="gptq2bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[compress,unlearn] wbits=3 tag="gptq3bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[compress,unlearn] wbits=4 tag="gptq4bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=gptq unlearn=ga interventions=[compress,unlearn] wbits=8 tag="gptq8bit-ga"
# # AWQ
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=2 tag="awq2bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=3 tag="awq3bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=4 tag="awq4bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=5 tag="awq5bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=6 tag="awq6bit-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=awq unlearn=ga interventions=[compress,unlearn] wbits=8 tag="awq8bit-ga"
# # Wanda
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.25 tag="wanda0.25\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.35 tag="wanda0.35\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.45 tag="wanda0.45\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.55 tag="wanda0.55\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.65 tag="wanda0.65\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.75 tag="wanda0.75\%-ga"
# # SparseGPT
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.25 tag="sparsegpt0.25\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.35 tag="sparsegpt0.35\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.45 tag="sparsegpt0.45\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.55 tag="sparsegpt0.55\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.65 tag="sparsegpt0.65\%-ga"
sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=ga interventions=[compress,unlearn] sparsity_ratio=0.75 tag="sparsegpt0.75\%-ga"