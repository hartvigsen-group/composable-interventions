#！/user/bin
python main.py \
    --model decapoda-research/llama-7b-hf --dataset c4 --wbits 4 --true-sequential --act-order --new-eval \
    --save out/llama_7b/un \
    --quant_method swq \
    --method quant 
