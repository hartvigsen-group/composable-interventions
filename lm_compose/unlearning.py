import copy
import json
import os
import random
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils.intervention_utils import get_dtype

from .easyeditor import ModelEditWrapper
from .wmdp.rmu import unlearn as rmu_unlearn
from .wmdp.rmu import utils as rmu_utils


def apply_ga(model, config, include_retain_loss=False):
    is_wrapper = isinstance(model, ModelEditWrapper)
    if is_wrapper:
        model = model.model

    # RMU only supports bfloat16
    ga_dtype = get_dtype("ga")
    if model.dtype != ga_dtype:
        print(f"GA: Converting model from {model.dtype} to {ga_dtype}")
        model = model.to(ga_dtype)

    # Freeze the first N layers of the transformer
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False

    N = 16
    for i in range(N):
        for param in model.model.layers[i].parameters():
            param.requires_grad = False

    # Make sure the final layers remain trainable
    # Note: Adjust the indexing based on your model's architecture
    for param in model.model.layers[N:].parameters():
        param.requires_grad = True

    # Also ensure the output layer remains trainable if present
    for param in model.lm_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.ga_lr)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    # Get unlearning target
    ascent_method_name = "Gradient Difference" if include_retain_loss else "Gradient Ascent"
    print(f"Loading {ascent_method_name} Datasets")
    ga_forget_set, ga_retain_set = get_ga_data(config.ga_forget_corpora, config.ga_retain_corpora, tokenizer)
    if config.ga_train_size:
        ga_forget_set.data = ga_forget_set.data[: config.ga_train_size]
        ga_retain_set.data = ga_retain_set.data[: config.ga_train_size]

    forget_dataloader = torch.utils.data.DataLoader(ga_forget_set, batch_size=config.ga_batch_size)
    retain_dataloader = torch.utils.data.DataLoader(ga_retain_set, batch_size=config.ga_batch_size)

    if include_retain_loss and config.ga_retain_weight != 1:
        print(f"Gradient Difference Retain Weight: {config.ga_retain_weight}")

    # Train model
    for epoch in range(config.ga_epochs):
        print(f"Epoch {epoch + 1}/{config.ga_epochs}")
        description = f"Training {ascent_method_name}"
        for batch_index, (forget_batch, retain_batch) in tqdm(
            enumerate(zip(forget_dataloader, retain_dataloader)),
            total=len(forget_dataloader),
            desc=description,
        ):
            forget_inputs = tokenizer(
                forget_batch,
                padding="max_length",
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(model.device)
            forget_inputs["labels"] = forget_inputs["input_ids"].clone()
            forget_outputs = model(**forget_inputs)
            forget_loss = (forget_outputs.loss * -1) / config.ga_grad_accumulation_steps
            batch_loss = forget_loss.clone()

            if include_retain_loss:
                retain_inputs = tokenizer(
                    retain_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                ).to(model.device)
                retain_inputs["labels"] = retain_inputs["input_ids"].clone()
                retain_outputs = model(**retain_inputs)
                retain_loss = config.ga_retain_weight * (retain_outputs.loss) / config.ga_grad_accumulation_steps
                batch_loss = batch_loss + retain_loss
                print(f"Batch Loss: {batch_loss.item()} Forget Loss: {forget_loss.item()} Retain Loss: {retain_loss.item()}")
            else:
                print(f"Batch Loss: {batch_loss.item()}")

            batch_loss.backward()
            if (batch_index + 1) % config.ga_grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

    # Prepare model for inference
    model.eval()

    # Cast back to configured dtype
    config_type = get_dtype(config.dtype)
    if model.dtype != config_type:
        print(f"Converting model to {config_type}")
        model = model.to(config_type)

    return model


class GADataset(Dataset):
    def __init__(self, data, tokenizer, min_len=50, max_len=2000):
        self.data = data
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokenize the text. The objective is next tokem prediction
        return self.data[idx]

    def collate_fn(self, batch):
        inputs = self.tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return inputs


def get_ga_data(forget_corpora, retain_corpora, tokenizer, min_len=50, max_len=2000, batch_size=1):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset

            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(str(x["text"]))
        else:
            current_dir_path = os.path.dirname(os.path.realpath(__file__))
            for line in open(f"{current_dir_path}/wmdp/data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)["text"]
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    # combine lists of datasets
    raw_forget_set = [get_dataset(c) for c in forget_corpora]
    raw_flatend_forget_set = []
    dataset_index = 0
    while True:
        pass_complet = dataset_index >= len(raw_forget_set[0]) and dataset_index >= len(raw_forget_set[1])
        if pass_complet:
            break
        if dataset_index < len(raw_forget_set[0]):
            raw_flatend_forget_set.append(raw_forget_set[0][dataset_index][0])
        if dataset_index < len(raw_forget_set[1]):
            raw_flatend_forget_set.append(raw_forget_set[1][dataset_index][0])

        dataset_index += 1

    # raw_forget_set = [item[0] for sublist in raw_forget_set for item in sublist]
    # random.shuffle(raw_forget_set)
    forget_set = GADataset(raw_flatend_forget_set, tokenizer, min_len=min_len, max_len=max_len)

    raw_retain_set = [get_dataset(c) for c in retain_corpora]
    if set(retain_corpora) == {"wikitext"}:
        raw_retain_set = [item[0] for item in raw_retain_set[0]]
    else:
        raw_retain_set = [item[0] for sublist in raw_retain_set for item in sublist]

    random.shuffle(raw_retain_set)
    retain_set = GADataset(raw_retain_set, tokenizer, min_len=min_len, max_len=max_len)

    return forget_set, retain_set


def get_layer_device(model, layer_name):
    try:
        layer = dict(model.named_modules())[layer_name]
        return next(layer.parameters()).device
    except (StopIteration, KeyError):
        return None


def distribute_model_across_devices(source_model, target_model):
    # Get the device map of the source model
    device_map = {name: get_layer_device(source_model, name) for name, _ in source_model.named_modules()}

    # Remove None values (layers without parameters or not found)
    device_map = {k: v for k, v in device_map.items() if v is not None}

    # Move each layer of the target model to the corresponding device
    for name, module in target_model.named_modules():
        if name in device_map:
            module.to(device_map[name])

    return target_model


def move_module_to_device(module, device):
    module.to(device)
    for param in module.parameters():
        param.data = param.data.to(device)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(device)


def copy_model_with_state_dict_and_device_map(original_model, device_map):
    # Create a new instance of the model
    new_model = copy.deepcopy(original_model)

    # Get the state dict from the original model
    state_dict = original_model.state_dict()

    # Load the state dict into the new model
    new_model.load_state_dict(state_dict)

    # Apply the device map
    for name, module in new_model.named_modules():
        if name in device_map and device_map[name] > 0:
            move_module_to_device(module, device_map[name])

    return new_model


def load_model_from_state_dict(model_class, state_dict, config):
    # Initialize the model
    model = model_class.from_config(config) if config else model_class()

    # Load the state dict into the model
    model.load_state_dict(state_dict)

    return model


def apply_rmu(model, config):
    """Unlearn WMDB Bio & Cyber with Representation Misdirection Unlearning (RMU)"""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    # RMU only supports bfloat16
    model = model.to(get_dtype("rmu"))

    is_wrapper = isinstance(model, ModelEditWrapper)
    frozen_copy_model = AutoModelForCausalLM.from_config(model.config)
    frozen_copy_model.load_state_dict(model.state_dict())
    for name, module in frozen_copy_model.named_modules():
        if name in model.hf_device_map:
            module.to(model.hf_device_map[name])

    frozen_copy_model = frozen_copy_model.to(get_dtype("rmu"))

    rmu_config = {
        "model_name_or_path": config.model_name,
        "module_str": "{model_name}.model.layers[{layer_id}]",
        "output_dir": None,
        "retain_corpora": config.rmu_retain_corpora,
        "forget_corpora": config.rmu_forget_corpora,
        "alpha": config.rmu_alpha,
        "steering_coeffs": config.rmu_steering_coeffs,
        "lr": config.rmu_lr,
        "min_len": config.rmu_min_len,
        "max_len": config.rmu_max_len,
        "batch_size": config.rmu_batch_size,
        "max_num_batches": 1000,
        "layer_id": config.rmu_layer_id,
        "layer_ids": [
            config.rmu_layer_id - 2,
            config.rmu_layer_id - 1,
            config.rmu_layer_id,
        ],
        "param_ids": [config.rmu_layer_id],
        "seed": config.rmu_seed,
        "verbose": True,
    }
    forget_data_list, retain_data_list = rmu_utils.get_data(
        rmu_config["forget_corpora"],
        rmu_config["retain_corpora"],
        rmu_config["min_len"],
        rmu_config["max_len"],
        rmu_config["batch_size"],
    )
    unlearned_model = rmu_unlearn.run_rmu(
        updated_model=model.model if is_wrapper else model,
        frozen_model=frozen_copy_model,
        tokenizer=tokenizer,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
        args=SimpleNamespace(**rmu_config),
    )

    # Cast back to configured dtype
    config_type = get_dtype(config.dtype)
    if unlearned_model.dtype != config_type:
        unlearned_model = unlearned_model.to(config_type)

    # Clean up VRAM for original model
    frozen_copy_model = frozen_copy_model.cpu()
    del frozen_copy_model
    torch.cuda.empty_cache()

    return model
