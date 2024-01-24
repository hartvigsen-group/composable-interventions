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


@hydra.main(version_base=None, config_path="conf", config_name="config_memit")
def main(config):
    hparams=config
    args=config
    # Create a timestamp
    timestamp = save_ckpt_meta.get_timestamp()

    # Initialize W&B (Remove layer list since it can't handle lists)
    config_dict = OmegaConf.to_container(config, resolve=True) # Convert the DictConfig to a standard Python dictionary
    config_dict.pop('layers', None) # Remove the 'layers' key
    wandb.init(
        project="counterfact_extra",
        config=config_dict,
        mode="online", # "disabled" for dry-runs, "online" for logging
        tags=[config.tag] # List of tags
    )

    if config.edit_train:
        # edit methods that requires training extra modules
        from easyeditor import ZsreDataset
        from easyeditor import EditTrainer
        from easyeditor import SERACTrainingHparams, MENDTrainingHparams
        if config.alg_name =='SERAC':
            training_hparams = SERACTrainingHparams.from_hparams(hparams.edit_train_config)
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

    # Get edits to be made
    prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs = edit_generator.get_edits(dataset=config.edit_dataset, number_of_edits=config.number_of_edits, edit_set=config.edit_set)

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
        pruning_and_validation = LLMPruningAndValidation(args, model)

        # Prune
        if config.compress and config.method == 'prune':
            pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
            pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

        # Quant
        if config.compress and config.method == 'quant':
            pruning_and_validation.quantization()
            model.to(f'cuda:{hparams.device}')
    
    # Make editable
    editable_model = ModelEditWrapper(model, hparams)

    if config.edit:
        editable_model.batch_edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            # rephrase_prompts=rephrase_prompt,
            # locality_inputs=locality_inputs,
            keep_original_weight=False
        )
        for p in editable_model.model.parameters():
            p.requires_grad_()
            

    if config.alg_name =='SERAC':
        print("warning! serac does not support the LLMPruningAndValidation with some bugs!")
    
    # Sparsify editable model
    pruning_and_validation = LLMPruningAndValidation(args, editable_model.model)

    # Prune
    if config.compress and config.method == 'prune':
        pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
        pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

    # Quant
    if config.compress and config.method == 'quant':
        pruning_and_validation.quantization()
        model.to(f'cuda:{hparams.device}')

    # Calculate and log eval metrics
    success_score, success_recall = evals.f1_accuracy_generate(model, prompts, target_new, config)
    generalization_score, gen_recall = evals.f1_accuracy_generate(model, rephrase_prompt, target_new, config)
    wandb.run.summary["Rewrite accuracy"] = success_score
    wandb.run.summary["Generalization"] = generalization_score


    if config.edit_dataset == "mquake":  # a hacky way to smuggle the mquake single hop prompts as "locality inputs"
        locality_score, local_recall = evals.f1_accuracy_generate(model, locality_inputs[0], locality_inputs[1], config)
        wandb.run.summary["Locality"] = locality_score
    else:
        locality_score, local_recall = evals.f1_locality_generate(model, locality_inputs, config)
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
    ppl_edits = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=True)
    ppl_edits_unmasked = evals.ppl_responses(model, prompts, target_new, config, mask_prompt=False)
    avgbits = pruning_and_validation.average_bits()
    pruning_and_validation.sparsity_check()
    if args.method != 'quant' or args.compress == False:
        flops = pruning_and_validation.FLOPs()
    else: flops = -1
    if args.method == 'quant' or args.compress == False:
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
    "Success recall": success_recall,
    "Generalization recall": gen_recall,
    "Local recall": local_recall
    })

    # Save checkpoint and metadata
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, timestamp, '/scratch/sux7mp/saved_models/')

if __name__ == '__main__':
    main()
