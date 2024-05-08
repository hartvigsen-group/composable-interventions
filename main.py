from main_quantize import LLMPruningAndValidation
from sparsellm.lib.prune import AverageBits
from easyeditor import MEMITHyperParams
from easyeditor import BaseEditor, ModelEditWrapper
import argparse
import os 
import numpy as np
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


@hydra.main(version_base=None, config_path="conf", config_name="config_memit")
def main(config):
    hparams=config
    config.dataset = config.compression_dataset # hacky way to smuggle the dataset name into the config

    # Create a timestamp
    timestamp = save_ckpt_meta.get_timestamp()

    # Initialize W&B (Remove layer list since it can't handle lists)
    config_dict = OmegaConf.to_container(config, resolve=True) # Convert the DictConfig to a standard Python dictionary
    config_dict.pop('layers', None) # Remove the 'layers' key
    wandb.init(
        project="Composable_Interventions",
        config=config_dict,
        mode="online", # "disabled" for dry-runs, "online" for logging
        tags=[config.tag] # List of tags
    )

    if config.edit_train:
        print('Starting editor training...')
        # edit methods that requires training extra modules
        from easyeditor import ZsreDataset
        from easyeditor import EditTrainer
        from easyeditor import SERACTrainingHparams, MENDTrainingHparams
        if config.alg_name =='SERAC':
            # training_hparams = SERACTrainingHparams.from_hparams(hparams.edit_train_config)
            training_hparams = config.edit_train_config
        elif config.alg_name =='MEND':
            training_hparams = MENDTrainingHparams.from_hparams(hparams.edit_train_config)
        print("warning! we need to decide the dataset to use for training serac and mend")
        train_ds = ZsreDataset('./data/zsre/zsre_mend_train.json', config=training_hparams)
        eval_ds = ZsreDataset('./data/zsre/zsre_mend_eval.json', config=training_hparams)
        trainer = EditTrainer(
            config=training_hparams,
            train_set=train_ds,
            val_set=eval_ds
        )
        trainer.run()
        print('Editor training complete.')

    # Init model
    model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                device_map="auto"
            )
    
    if config.load_ckpt:
        # Load the state_dict
        state_dict = torch.load(config.ckpt_path)

        # Update the model's state_dict
        model.load_state_dict(state_dict)

    if config.compress_first:
        # Sparsify editable model
        pruning_and_validation = LLMPruningAndValidation(hparams, model)

        # Prune
        if config.compress and config.method == 'prune':
            pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
            pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

        # Quant
        if config.compress and config.method == 'quant':
            pruning_and_validation.quantization()
            model.to(f'cuda:{hparams.device}')

    # Apply unlearning to the model
    if config.unlearn:
        if config.unlearn_method == "rmu":
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, trust_remote_code=True, use_fast=False
            )
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"
            tokenizer.mask_token_id = tokenizer.eos_token_id
            tokenizer.sep_token_id = tokenizer.eos_token_id
            tokenizer.cls_token_id = tokenizer.eos_token_id
            
            unlearning_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.bfloat16, 
                # low_cpu_mem_usage=True, 
                device_map="auto"
            )
            rmu_config = {
                "model_name_or_path": config.model_name,
                # "module_str": f"{config.model_name}.model.layers[{config.rmu_layer_id}]",
                "module_str": "{model_name}.model.layers[{layer_id}]",
                "output_dir": None,
                "retain_corpora": config.rmu_retain_corpora,
                "forget_corpora": config.rmu_forget_corpora,
                "alpha": config.rmu_alpha,
                "steering_coeff_list": config.rmu_steering_coeff_list,
                "lr": config.rmu_lr,
                "min_len": config.rmu_min_len,
                "max_len": config.rmu_max_len,
                "batch_size": config.rmu_batch_size,
                "max_num_batches": config.rmu_max_num_batches,
                "layer_id": config.rmu_layer_id,
                "layer_ids": config.rmu_layer_ids,
                "param_ids": config.rmu_param_ids,
                "seed": config.rmu_seed
            }
            forget_data_list, retain_data_list = rmu_utils.get_data(
                rmu_config["forget_corpora"],
                rmu_config["retain_corpora"],
                rmu_config["min_len"],
                rmu_config["max_len"],
                rmu_config["batch_size"],
            )

            # Updates unlearning_model
            rmu_unlearn.run_rmu(
                updated_model=unlearning_model,
                frozen_model=model,
                tokenizer=tokenizer,
                forget_data_list=retain_data_list,
                retain_data_list=retain_data_list,
                args=rmu_config
            )
    
        elif config.unlearn_method == "gradient_ascent":
            raise NotImplementedError("Gradient ascent hasn't been implemented yet")
        else:
            raise NotImplementedError(f"Unlearning method not supported: {config.unlearn_method}")
    
    # Make editable
    editable_model = ModelEditWrapper(model, hparams)
    device_map = editable_model.model.hf_device_map

    # Get edits to be made
    prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs = edit_generator.get_edits(dataset=config.edit_dataset, number_of_edits=config.number_of_edits, edit_set=config.edit_set)

    # print(model)
    if config.edit:
        editable_model.batch_edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            keep_original_weight=False
        )
        for p in editable_model.model.parameters():
            p.requires_grad_()
        print('editing complete')
    editable_model.model.hf_device_map = device_map

    if config.alg_name =='SERAC':
        # print("warning! serac does not support the LLMPruningAndValidation with some bugs!")
        pruning_and_validation = LLMPruningAndValidation(args, model.model)
    else:
        # Sparsify editable model
        pruning_and_validation = LLMPruningAndValidation(hparams, editable_model.model)

    # Prune
    if config.compress and config.method == 'prune':
        pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
        pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

    # Quant
    if config.compress and config.method == 'quant':
        pruning_and_validation.pseudoQuantization()
        editable_model.to(f'cuda:{hparams.device}')

    # Save checkpoint and metadata
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, timestamp, '/scratch/sux7mp/saved_models/')
    
    # Begin evaluations
    print("Starting eval...")

    # Evaluate on QA benchmarks
    print(f"Evaluating QA benchmarks...")
    lm_eval_model = HFLM(model)
    task_manager = lm_eval.tasks.TaskManager()
    qa_benchmarks = ["mmlu", "wmdp_cyber", "wmdp_bio"] if config.unlearn else ["mmlu"]
    qa_benchmark_results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm_eval_model,
        tasks=qa_benchmarks,
        num_fewshot=0,
        task_manager=task_manager,
        # limit=5
    )

    for benchmark_name in qa_benchmark_results["groups"]:
        benchmark_accuracy = qa_benchmark_results["groups"][benchmark_name]["acc,none"]
        benchmark_std_error = qa_benchmark_results["groups"][benchmark_name]["acc_stderr,none"]
        wandb.run.summary["{benchmark_name} accuracy"] = benchmark_accuracy
        wandb.run.summary["{benchmark_name} stderr"] = benchmark_std_error
        print(f"{benchmark_name} - Accuracy: {benchmark_accuracy} StdErr: {benchmark_std_error}")
    
    print("Starting editing eval...")
    success_score, success_recall = evals.f1_accuracy_generate(editable_model, prompts, target_new, config)
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
        latency = pruning_and_validation.CalculateLatency()
    else: latency = -1

    # Save to WandB
    wandb.run.summary["PPL"] = ppl_test
    wandb.run.summary["Average bits"] = avgbits
    wandb.run.summary["FLOPs"] = flops
    wandb.run.summary["Latency"] = latency

    wandb.log({
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
    })

if __name__ == '__main__':
    main()
