#!/bin/bash

### RMU only ###
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=none unlearn=rmu interventions=[unlearn] tag="rmu-none"

# ### RMU then Compress ###
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-rightarrow-wanda0.25\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-rightarrow-wanda0.45\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-rightarrow-wanda0.65\%"
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.25 tag="rmu-rightarrow-sparsegpt0.25\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.45 tag="rmu-rightarrow-sparsegpt0.45\%"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[unlearn,compress] sparsity_ratio=0.65 tag="rmu-rightarrow-sparsegpt0.65\%"

# ### Compress then RMU ###
# # Wanda
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="wanda0.25\%-rightarrow-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="wanda0.45\%-rightarrow-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=wanda unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="wanda0.65\%-rightarrow-rmu"
# # SparseGPT
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.25 tag="sparsegpt0.25\%-rightarrow-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.45 tag="sparsegpt0.45\%-rightarrow-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=none compression=sparsegpt unlearn=rmu interventions=[compress,unlearn] sparsity_ratio=0.65 tag="sparsegpt0.65\%-rightarrow-rmu"

### RMU then Edit ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-rightarrow-lora"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-rightarrow-ft"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit compression=none unlearn=rmu interventions=[unlearn,edit] tag="rmu-rightarrow-memit"

### Edit then RMU ###
sbatch run_exp.sh save_ckpt=False wandb=online edit=lora compression=none unlearn=rmu interventions=[edit,unlearn] tag="lora-rightarrow-rmu"
# sbatch run_exp.sh save_ckpt=False wandb=online edit=ft compression=none unlearn=rmu interventions=[edit,unlearn] tag="ft-rightarrow-rmu"
sbatch run_exp.sh save_ckpt=False wandb=online edit=memit compression=none unlearn=rmu interventions=[edit,unlearn] tag="memit-rightarrow-rmu"