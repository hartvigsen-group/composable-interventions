import os
import random
import sys

import hydra
import lm_eval
import numpy as np
import pandas as pd
import torch
import wandb
from lm_eval.models.huggingface import HFLM
from omegaconf import OmegaConf
from tabulate import tabulate
from transformers import AutoModelForCausalLM
from unlearning import apply_ga, apply_rmu
from utils import edit_generator, evals, save_ckpt_meta
from utils.intervention_utils import get_dtype

from .easyeditor import ModelEditWrapper
from .main_quantize import LLMPruningAndValidation


def edit_model(model, config, prompts, ground_truth, target_new, subject):
    model = model.to(dtype=get_dtype(config.edit))
    editable_model = ModelEditWrapper(model, config)
    if config.alg_name != "LoRA":
        editable_model.train()
    editable_model.batch_edit(prompts=prompts, ground_truth=ground_truth, target_new=target_new, subject=subject, keep_original_weight=False)
    if config.alg_name == "LoRA":
        editable_model = editable_model.merge_and_unload()
    for p in editable_model.model.parameters():
        p.requires_grad_()
    return editable_model


def compress_model(model, config, pruning_and_validation):
    if config.method == "quant":
        model = model.to(dtype=get_dtype(config.compression))
        
        # Clean up model?
        del model
        torch.cuda.empty_cache()

        pruning_and_validation.pseudoQuantization()
        model = pruning_and_validation.model
        model.to(f"cuda:{config.device}")

        del pruning_and_validation
        pruning_and_validation = LLMPruningAndValidation(config, model)
        torch.cuda.empty_cache()
        return model
    elif config.method == "prune":
        model = model.to(dtype=get_dtype(config.compression))
        pruning_and_validation = LLMPruningAndValidation(config, model)
        pruning_and_validation.get_Mask()  # Obtain mask once
        pruning_and_validation.prune()  # Apply pruning
        return model
    else:
        raise ValueError(f"Invalid compression method: {config.method}")


def unlearn_model(model, config):
    if config.unlearn_method == "rmu":
        return apply_rmu(model, config)
    if config.unlearn_method == "ga":
        return apply_ga(model, config, include_retain_loss=False)
    if config.unlearn_method == "gd":
        return apply_ga(model, config, include_retain_loss=True)

    raise ValueError(f"Invalid unlearn method: {config.unlearn_method}")


def get_qa_results(model, config):
    lm_eval_model = HFLM(model)
    task_manager = lm_eval.tasks.TaskManager()
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


def format_config(config):
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
    sections_to_flatten = ["edit", "compression", "unlearn"]
    for section in sections_to_flatten:
        if section in config:
            for key, value in config[section].items():
                config[key] = value

    # Apply command line overrides after flattening the configuration
    return OmegaConf.merge(config, OmegaConf.create(filtered_overrides))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    config = format_config(config)

    hparams = config.copy()
    config.dataset = config.compression_dataset  # hacky way to smuggle the dataset name into the config

    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Create a timestamp
    timestamp = save_ckpt_meta.get_timestamp()

    # Initialize W&B (Remove layer list since it can't handle lists)
    config_dict = OmegaConf.to_container(config, resolve=True)  # Convert the DictConfig to a standard Python dictionary
    config_dict.pop("layers", None)  # Remove the 'layers' key
    experiment_id = f"{config.tag}-{timestamp}"
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=experiment_id,
        config=config_dict,
        mode=config.wandb,  # "disabled" for dry-runs, "online" for logging
        tags=[config.tag],  # List of tags
    )

    # Init model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=get_dtype(config.dtype), device_map="balanced")

    # Make editable
    editable_model = ModelEditWrapper(model, hparams)
    device_map = editable_model.model.hf_device_map

    # Strange bug where config.device becomes a list somewhere. Cast back to an int.
    if not isinstance(config.device, int) and len(config.device) == 2 and config.device[0] == "cuda":
        print("Resetting config.device")
        config.deviswsvece = int(config.device[-1])

    if not isinstance(hparams.device, int) and len(hparams.device) == 2 and hparams.device[0] == "cuda":
        print("Resetting hparams.device")
        hparams.device = int(hparams.device[-1])

    # Get edits to be made
    prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs = edit_generator.get_edits(
        dataset=config.edit_dataset, number_of_edits=config.number_of_edits, edit_set=config.edit_set
    )

    # Use LLMPruningAndValidation for handling compression
    pruning_and_validation = LLMPruningAndValidation(config, model)

    if config.load_ckpt:
        # Load the state_dict
        state_dict = torch.load(config.ckpt_path)

        # Update the model's state_dict
        model.load_state_dict(state_dict)

    # Check if the first operation in the initial list is compression-related
    is_multiple_interventions = len(config.interventions) > 1
    is_compress_first = (
        is_multiple_interventions and config.interventions[0] in ["compress", "compression", "quant", "prune"] and config.method in ["quant", "prune"]
    )
    is_not_awq = config.method != "quant" or config.quant_method != "autoawq"
    if is_multiple_interventions and is_compress_first and is_not_awq:
        # Append the first operation to the end of the list if it's compression-related to make sure final model is
        # compressed (not compression-aware editing)
        config.interventions.append(config.interventions[0])
        print(f"Appended {config.interventions[0]} to the end of the list to ensure final model is compressed")

    for intervention in config.interventions:
        print(f"############# Begin intervention: {intervention} #############")
        if intervention == "edit":
            model = edit_model(model, config, prompts, ground_truth, target_new, subject)
            editable_model.model.hf_device_map = device_map
        elif intervention in {"compress", "compression", "prune", "quant"}:
            model = compress_model(model, config, pruning_and_validation)
        elif intervention == "unlearn":
            model = unlearn_model(model, config)
            editable_model.model.hf_device_map = device_map
        else:
            raise ValueError(f"Invalid intervention: {intervention}")

    # Save checkpoint and metadata
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, timestamp, "/scratch/sux7mp/saved_models/")

    # Begin evaluations
    print("Starting eval...")
    print("Evaluating QA benchmarks...")
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
    ppl_test = pruning_and_validation.validate()  # It is a validation for general performance on common language benchmark such as wikitext.
    print("Starting PPL edit evals...")
    ppl_edits = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=True)
    ppl_edits_unmasked = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=False)
    ppl_QA = evals.ppl_QA(model, config)

    print("Starting Avg bits eval...")
    avgbits = pruning_and_validation.average_bits()

    # pruning_and_validation.sparsity_check()
    if hparams.method != "quant" or hparams.compress is False:
        print("Starting FLOPs eval...")
        flops = pruning_and_validation.FLOPs()
    else:
        flops = -1
    if hparams.method == "quant" or hparams.compress is False:
        print("Starting latency eval...")
        latency = pruning_and_validation.CalculateLatency(model)
    else:
        latency = -1

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
        "Local recall": local_recall,
    }
    wandb_log.update(qa_results)
    wandb.log(wandb_log)
    wanda_log_frame = pd.DataFrame([wandb_log]).T
    print("\nExperiment Metrics")
    print(tabulate(wanda_log_frame, headers="keys", tablefmt="psql"))

    # Log table to W&B
    wandb.run.log({"Metrics": wandb.Table(dataframe=wanda_log_frame)})


if __name__ == "__main__":
    main()
