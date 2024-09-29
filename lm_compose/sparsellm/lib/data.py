# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import logging
import time


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    logging.info("Loading C4 dataset...")
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    logging.info(f"Loaded {len(traindata)} train samples and {len(valdata)} validation samples")

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    # for _ in range(nsamples):
    #     while True:
    #         i = random.randint(0, len(traindata) - 1)
    #         trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
    #         if trainenc.input_ids.shape[1] > seqlen:
    #             break
    #     i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
    #     j = i + seqlen
    #     inp = trainenc.input_ids[:, i:j]
    #     tar = inp.clone()
    #     tar[:, :-1] = -100
    #     trainloader.append((inp, tar))

    logging.info(f"Generating {nsamples} samples...")

    all_text = " ".join(traindata[:1000]['text'])  # Join first 1000 samples
    all_tokens = tokenizer.encode(all_text)
    
    for sample_idx in range(nsamples):
        start_time = time.time()
        
        if len(all_tokens) < seqlen:
            logging.warning(f"Not enough tokens to generate sample {sample_idx}. Tokenized text length: {len(all_tokens)}")
            break
        
        start_idx = random.randint(0, len(all_tokens) - seqlen)
        inp = all_tokens[start_idx:start_idx + seqlen]
        tar = inp.copy()
        tar[:-1] = [-100] * (len(tar) - 1)
        
        # Convert to tensors
        inp_tensor = torch.tensor(inp).unsqueeze(0)  # Add batch dimension
        tar_tensor = torch.tensor(tar).unsqueeze(0)
        
        trainloader.append((inp_tensor, tar_tensor))
        end_time = time.time()
        logging.info(f"Sample {sample_idx} processed in {end_time - start_time:.2f} seconds")

    # Prepare validation dataset
    # valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    # valenc = valenc.input_ids[:, : (256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)

    logging.info("Preparing validation dataset...")
    try:
        val_text = " ".join(valdata[:1100]['text'])
        valenc = tokenizer.encode(val_text)
        valenc = valenc[:(256 * seqlen)]
        valenc = torch.tensor(valenc).unsqueeze(0)  # Add batch dimension
        valenc = TokenizerWrapper(valenc)
        logging.info("Validation dataset prepared successfully")
    except Exception as e:
        logging.error(f"Error preparing validation dataset: {str(e)}")
        valenc = None

    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=8192, tokenizer=None):
    print(f"Using {name}")
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
