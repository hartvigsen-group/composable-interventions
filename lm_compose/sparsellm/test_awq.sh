#！/user/bin
python main.py \
    --model yahma/llama-7b-hf --dataset wikitext --wbits 4 --true-sequential --act-order --new-eval \
    --save out/llama_7b/unstructured/wanda/ \
    --save_model out/ \
    --method quant \
    --groupsize 128 \
    --quant_method awq
