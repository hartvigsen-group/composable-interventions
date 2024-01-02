from sparsellm.main import LLMPruningAndValidation
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
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    hparams=config
    args=config
    # Create a timestamp
    timestamp = save_ckpt_meta.get_timestamp()

    # Initialize a writer
    writer = SummaryWriter(log_dir=f'runs/{timestamp}')

    # Get edits to be made
    prompts, ground_truth, target_new, subject, rephrase_prompt, locality_inputs = edit_generator.get_edits(number_of_edits=config.number_of_edits)

    
    # Init model
    model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
                device_map="auto"
            )

    if config.load_ckpt:
        # Load the state_dict
        state_dict = torch.load(config.ckpt_path)

        # Update the model's state_dict
        model.load_state_dict(state_dict)

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

    # Sparsify editable model
    pruning_and_validation = LLMPruningAndValidation(args, editable_model.model)

    # Prune
    if config.compress and config.method == 'prune':
        pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
        pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

    # Quant
    if config.compress and config.method == 'quant':
        pruning_and_validation.quantization()

    # Calculate eval metrics
    success_score = evals.calculate_edit_accuracy_logits(model, prompts, target_new, config)
    locality_score = evals.F1_locality_generate(model, locality_inputs, config)
    generalization_score = evals.calculate_edit_accuracy(model, rephrase_prompt, target_new, config)
    writer.add_scalar("Rewrite accuracy", success_score, 1)
    writer.add_scalar("Locality", locality_score, 1)
    writer.add_scalar("Generalization", generalization_score, 1)

    # Print eval metrics
    print(f"Success: {success_score}")
    print(f"Locality: {locality_score}")
    print(f"Generalization: {generalization_score}")

    # Validate ppl
    ppl_test = pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
    writer.add_scalar("PPL", ppl_test, 1)

    # Save the hparams and metrics to tensorboard (Remove layer list since it can't handle lists)
    config_dict = OmegaConf.to_container(config, resolve=True) # Convert the DictConfig to a standard Python dictionary
    config_dict.pop('layers', None) # Remove the 'layers' key
    writer.add_hparams(config_dict, {"Rewrite accuracy": success_score, "Locality": locality_score, "Generalization": generalization_score, "PPL": ppl_test})

    # Save checkpoint and metadata
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, timestamp, '/scratch/sux7mp/saved_models/')

if __name__ == '__main__':
    main()
