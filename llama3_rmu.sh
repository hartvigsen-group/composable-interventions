python -m rmu.unlearn \ 
    --retain_corpora=wikitext \
    --forget_corpora=cyber-forget-corpus \
    --model_name_or_path=meta-llama/Meta-Llama-3-8B \
    --alpha=$alpha \ 
    --max_num_batche=500 --layer_id=$layer --layer_ids=$(($layer-1)),$layer,$(($layer+1)) --lr=$lr --no_save"