#！/user/bin
python main.py \
    --model decapoda-research/llama-7b-hf \
    --method quant \
    --zero_point \
    --prune_method wanda \
    --sparsity_ratio 0.7 \
    --sparsity_type unstructured \
    --save out/llama_7b/unstructured/wanda/ 
