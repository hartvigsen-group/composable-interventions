#!/bin/bash

# Define different sets of parameters
params=(
    ### None ###
    'edit=False compress=False compress_first=False sparsity_ratio=0.0 tag=none'
    ### Edit only ###
    'edit=True compress=False compress_first=False sparsity_ratio=0.0 tag=MEMIT'
    ### Compress only ###
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.25 tag=Wanda$_\{25\%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.35 tag=Wanda$_\{35%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.45 tag=Wanda$_\{45%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.55 tag=Wanda$_\{55%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.65 tag=Wanda$_\{65%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.75 tag=Wanda$_\{75%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.85 tag=Wanda$_\{85%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.25 tag=SparseGPT$_\{25%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.35 tag=SparseGPT$_\{35%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.45 tag=SparseGPT$_\{45%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.55 tag=SparseGPT$_\{55%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.65 tag=SparseGPT$_\{65%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.75 tag=SparseGPT$_\{75%\}$'
    'edit=False compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.85 tag=SparseGPT$_\{85%\}$'
    ### Edit then compress ###
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.25 tag=MEMIT-rightarrow-Wanda$_\{25%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.35 tag=MEMIT-rightarrow-Wanda$_\{35%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.45 tag=MEMIT-rightarrow-Wanda$_\{45%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.55 tag=MEMIT-rightarrow-Wanda$_\{55%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.65 tag=MEMIT-rightarrow-Wanda$_\{65%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.75 tag=MEMIT-rightarrow-Wanda$_\{75%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=wanda sparsity_ratio=0.85 tag=MEMIT-rightarrow-Wanda$_\{85%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.25 tag=MEMIT-rightarrow-SparseGPT$_\{25%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.35 tag=MEMIT-rightarrow-SparseGPT$_\{35%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.45 tag=MEMIT-rightarrow-SparseGPT$_\{45%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.55 tag=MEMIT-rightarrow-SparseGPT$_\{55%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.65 tag=MEMIT-rightarrow-SparseGPT$_\{65%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.75 tag=MEMIT-rightarrow-SparseGPT$_\{75%\}$'
    'edit=True compress=True compress_first=False method=prune prune_method=sparsegpt sparsity_ratio=0.85 tag=MEMIT-rightarrow-SparseGPT$_\{85%\}$'
    # ### Compress then edit (then compress) ###
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.25 tag=Wanda$_\{25%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.35 tag=Wanda$_\{35%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.45 tag=Wanda$_\{45%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.55 tag=Wanda$_\{55%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.65 tag=Wanda$_\{65%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.75 tag=Wanda$_\{75%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=wanda sparsity_ratio=0.85 tag=Wanda$_\{85%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.25 tag=SparseGPT$_\{25%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.35 tag=SparseGPT$_\{35%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.45 tag=SparseGPT$_\{45%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.55 tag=SparseGPT$_\{55%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.65 tag=SparseGPT$_\{65%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.75 tag=SparseGPT$_\{75%\}-rightarrow-MEMIT$'
    'edit=True compress=True compress_first=True method=prune prune_method=sparsegpt sparsity_ratio=0.85 tag=SparseGPT$_\{85%\}-rightarrow-MEMIT$'
)

# Loop through each set of parameters and submit a job
for p in "${params[@]}"; do
    sbatch run_exp10x.sh $p
done
