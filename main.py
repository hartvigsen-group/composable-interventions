from main_quantize import LLMPruningAndValidation
from sparsellm.lib.prune import AverageBits
from easyeditor import MEMITHyperParams
from easyeditor import BaseEditor, ModelEditWrapper
from tabulate import tabulate
import argparse
import os 
import sys
import copy
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import copy
import hashlib
import yaml
import hydra
from omegaconf import OmegaConf
from utils import edit_generator, save_ckpt_meta, evals
import wandb
from wmdp.rmu import unlearn as rmu_unlearn
from wmdp.rmu import utils as rmu_utils
import lm_eval
from lm_eval.models.huggingface import HFLM
from types import SimpleNamespace
import copy


def edit_model(model, config, prompts, ground_truth, target_new, subject):
    # Use ModelEditWrapper for handling edits
    model = model.to(dtype=get_dtype(config.edit))
    editable_model = ModelEditWrapper(model, config)
    if config.alg_name != 'LoRA':
        editable_model.train()
    editable_model.batch_edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        keep_original_weight=False
    )
    if config.alg_name == 'LoRA':
        editable_model = editable_model.merge_and_unload()
    for p in editable_model.model.parameters():
        p.requires_grad_()
    return editable_model


def compress_model(model, config, pruning_and_validation):
    del pruning_and_validation
    pruning_and_validation = LLMPruningAndValidation(config, model)
    if config.method == 'quant':
        model = model.to(dtype=get_dtype(config.compression))
        # Set any Nans to zero
        # for param in model.parameters():
        #     if param.requires_grad:
        #         param.data.masked_fill_(torch.isnan(param.data), 0)

        pruning_and_validation.pseudoQuantization()
        model.to(f'cuda:{config.device}')
        return model
    elif config.method == 'prune':
        model = model.to(dtype=get_dtype(config.compression))
        pruning_and_validation = LLMPruningAndValidation(config, model)
        pruning_and_validation.get_Mask()  # Obtain mask once
        pruning_and_validation.prune()     # Apply pruning
        return model
    else:
        raise ValueError(f"Invalid compression method: {config.method}")


def unlearn_model(model, config):
    """Unlearn WMDB Bio & Cyber with Representation Misdirection Unlearning (RMU)"""

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    # Cast model to CPU
    # model = model.cpu()

    # RMU only supports bfloat16
    model = model.to(get_dtype("rmu"))
    is_wrapper = isinstance(model, ModelEditWrapper)
    frozen_copy_model = copy.deepcopy(model.model) if is_wrapper else copy.deepcopy(model)
    # frozen_copy_model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=model.dtype, device_map="auto")
    # state_dict = model.model.state_dict() if is_wrapper else model.state_dict()
    # frozen_copy_model.load_state_dict(state_dict)

    # put model back on GPU
    # model = model.to(unlearning_model.device)

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
        "layer_ids": config.rmu_layer_ids,
        "param_ids": config.rmu_param_ids,
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
        args=SimpleNamespace(**rmu_config)
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


def get_qa_results(model, config):
    lm_eval_model = HFLM(model)
    task_manager = lm_eval.tasks.TaskManager()
    is_rmu_enabled = "unlearn" in config.interventions and config.unlearn_method == "rmu"
    # qa_benchmarks = ["mmlu", "wmdp_cyber", "wmdp_bio"] if is_rmu_enabled else ["mmlu"]
    qa_benchmarks = ["mmlu", "wmdp_cyber", "wmdp_bio"]
    qa_benchmark_results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=qa_benchmarks,
        num_fewshot=0,
        task_manager=task_manager,
        batch_size=16,
        limit=config.qa_question_count_limit,
    )
    
    benchmark_results = {}
    for benchmark_name in qa_benchmarks:
        benchmark_accuracy = qa_benchmark_results["results"][benchmark_name]["acc,none"]
        benchmark_std_error = qa_benchmark_results["results"][benchmark_name]["acc_stderr,none"]
        benchmark_results[benchmark_name] = benchmark_accuracy
        wandb.run.summary[f"{benchmark_name} accuracy"] = benchmark_accuracy
        wandb.run.summary[f"{benchmark_name} stderr"] = benchmark_std_error
        print(f"{benchmark_name} - Accuracy: {round(benchmark_accuracy, 2)} StdErr: {round(benchmark_std_error, 2)}")
    
    return benchmark_results


def get_dtype(dtype_str):

    
    """Dynamically get the torch dtype based on the config"""
    dtype_mapping = {
        'torch.float': torch.float,
        'torch.float32': torch.float32,
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.float64': torch.float64,  # Adding more possible dtypes
        'torch.half': torch.half,
        'torch.double': torch.double,
        'awq': torch.float16,
        'gptq': torch.float16,
        'wanda': torch.bfloat16,
        'sparsegpt': torch.bfloat16,
        'ft': torch.bfloat16,
        'memit': torch.bfloat16,
        'lora': torch.float,
        "rmu": torch.bfloat16,
    }
    
    if dtype_str not in dtype_mapping:
        raise ValueError(f"Invalid dtype specified in config: {dtype_str}")
    
    return dtype_mapping[dtype_str]

@hydra.main(version_base=None, config_path="conf", config_name="config")

def main(config):
    # To make sections backwards compatible with old code
    # Capture command line arguments
    command_line_args = sys.argv[1:]
    command_line_overrides = OmegaConf.from_dotlist(command_line_args)

    # Define Hydra's special arguments to exclude
    hydra_special_args = {"--multirun", "-m", "--run", "-r", "--config-path", "--config-name"}

    # Filter out Hydra's special arguments
    filtered_overrides = {k: v for k, v in command_line_overrides.items() if k not in hydra_special_args}

    # Temporarily disable strict structure enforcement
    OmegaConf.set_struct(config, False)

    # Dynamicaly set the corect user in config path
    for key, value in config.items():
        if isinstance(value, str) and "{USER}" in value:
            config[key] = value.replace("{USER}", os.environ["USER"])

    # Flatten the configuration
    sections_to_flatten = ['edit', 'compression', 'unlearn']
    for section in sections_to_flatten:
        if section in config:
            # Move each sub-configuration to the top level
            for key, value in config[section].items():
                config[key] = value
            # Optionally delete the original section
            # del config[section]

    # Apply command line overrides after flattening the configuration
    config = OmegaConf.merge(config, OmegaConf.create(filtered_overrides))

    hparams=config
    config.dataset = config.compression_dataset # hacky way to smuggle the dataset name into the config

    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create a timestamp
    timestamp = save_ckpt_meta.get_timestamp()

    # Initialize W&B (Remove layer list since it can't handle lists)
    config_dict = OmegaConf.to_container(config, resolve=True) # Convert the DictConfig to a standard Python dictionary
    config_dict.pop('layers', None) # Remove the 'layers' key
    experiment_id = f"{config.tag}-{timestamp}"
    wandb.init(
        project="Composable_Interventions",
        name=experiment_id,
        config=config_dict,
        mode=config.wandb, # "disabled" for dry-runs, "online" for logging
        tags=[config.tag] # List of tags
    )

    # Init model
    model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=get_dtype(config.dtype),
                # low_cpu_mem_usage=True, 
                device_map="auto"
            )

    # Make editable
    editable_model = ModelEditWrapper(model, hparams)
    device_map = editable_model.model.hf_device_map

    # Get edits to be made
    prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs = edit_generator.get_edits(dataset=config.edit_dataset, number_of_edits=config.number_of_edits, edit_set=config.edit_set)

    # Use LLMPruningAndValidation for handling compression
    pruning_and_validation = LLMPruningAndValidation(config, model)

    if config.load_ckpt:
        # Load the state_dict
        state_dict = torch.load(config.ckpt_path)

        # Update the model's state_dict
        model.load_state_dict(state_dict)

    # Check if the first operation in the initial list is compression-related
    is_multiple_interventions = len(config.interventions) > 1
    is_compress_first = is_multiple_interventions and config.interventions[0] in ['compress', 'compression', 'quant', 'prune'] and config.method in ['quant', 'prune']
    is_not_awq = config.method != "quant" or config.quant_method != "autoawq"
    if is_multiple_interventions and is_compress_first and is_not_awq:
        # Append the first operation to the end of the list if it's compression-related to make sure final model is compressed (not compression-aware editing)
        config.interventions.append(config.interventions[0])
        print(f"Appended {config.interventions[0]} to the end of the list to ensure final model is compressed")


    for intervention in config.interventions:
        print(f"############# Begin intervention: {intervention} #############")
        if intervention == 'edit':
            model = edit_model(model, config, prompts, ground_truth, target_new, subject)
            editable_model.model.hf_device_map = device_map
        elif intervention in {'compress', 'compression', 'prune', 'quant'}:
            model = compress_model(model, config, pruning_and_validation)
        elif intervention == 'unlearn':
            model = unlearn_model(model, config)
            editable_model.model.hf_device_map = device_map
        else:
            raise ValueError(f"Invalid intervention: {intervention}")
    
    # Save checkpoint and metadata
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, timestamp, '/scratch/sux7mp/saved_models/')
    
    # Begin evaluations
    print("Starting eval...")
    print(f"Evaluating QA benchmarks...")
    qa_results = get_qa_results(editable_model, config)
    
    print("Starting editing eval...")
    success_score, success_recall = evals.f1_accuracy_generate(editable_model, prompts, target_new, config, verbose=True)
    generalization_score, gen_recall = evals.f1_accuracy_generate(editable_model, rephrase_prompt, target_new, config)
    wandb.run.summary["Rewrite accuracy"] = success_score
    wandb.run.summary["Generalization"] = generalization_score

    if config.edit_dataset == "mquake":  # a hacky way to smuggle the mquake single hop prompts as "locality inputs"
        locality_score, local_recall = evals.f1_accuracy_generate(editable_model, locality_inputs[0], locality_inputs[1], config)
        wandb.run.summary["Locality"] = locality_score
    else:
        locality_score, local_recall = evals.f1_locality_generate(editable_model, locality_inputs, config)
        wandb.run.summary["Locality"] = locality_score

    # Print eval metrics
    print(f"Success: {success_score}")
    print(f"Generalization: {generalization_score}")
    print(f"Locality/one hop: {locality_score}")

    print(f"Success recall: {success_recall}")
    print(f"Generalization recall: {gen_recall}")
    print(f"Locality/one hop recall: {local_recall}")

    # Metrics and evaluation
    ppl_test = pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
    print('Starting PPL edit evals...')
    ppl_edits = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=True)
    ppl_edits_unmasked = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=False)
    ppl_QA = evals.ppl_QA(model, config)
    
    print('Starting Avg bits eval...')
    avgbits = pruning_and_validation.average_bits()
    
    # pruning_and_validation.sparsity_check()
    if hparams.method != 'quant' or hparams.compress == False:
        print('Starting FLOPs eval...')
        flops = pruning_and_validation.FLOPs()
    else: flops = -1
    if hparams.method == 'quant' or hparams.compress == False:
        print('Starting latency eval...')
        latency = pruning_and_validation.CalculateLatency(model)
    else: latency = -1

    # Save to WandB
    wandb.run.summary["PPL"] = ppl_test
    wandb.run.summary["Average bits"] = avgbits
    wandb.run.summary["FLOPs"] = flops
    wandb.run.summary["Latency"] = latency

    wandb_log = {
        "Rewrite accuracy": success_score,
        "Generalization": generalization_score,
        "Locality": locality_score,
        "PPL": ppl_test,
        "PPL edits": ppl_edits,
        "PPl edits unmasked": ppl_edits_unmasked,
        "PPl QA": ppl_QA,
        "Success recall": success_recall,
        "Generalization recall": gen_recall,
        "Local recall": local_recall
    }
    wandb_log.update(qa_results)
    wandb.log(wandb_log)
    wanda_log_frame = pd.DataFrame([wandb_log]).T
    print("\nExperiment Metrics")
    print(tabulate(wanda_log_frame, headers='keys', tablefmt='psql'))

    # Log table to W&B
    wandb.run.log({"Metrics": wandb.Table(dataframe=wanda_log_frame)})

if __name__ == '__main__':
    main()
