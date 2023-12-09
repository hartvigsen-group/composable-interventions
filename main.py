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
from utils import edit_generator, save_ckpt_meta, evals
from torch.utils.tensorboard import SummaryWriter


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config):
    hparams=config
    args=config

    # Initialize a writer
    writer = SummaryWriter()
    
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
        # Check initial edit metrics
        # metrics = editable_model.evaluate(
        #     model=editable_model,
        #     prompts=prompts,
        #     ground_truth=ground_truth,
        #     target_new=target_new,
        #     subject=subject,
        #     # rephrase_prompts=rephrase_prompt,
        #     locality_inputs=locality_inputs,
        #     keep_original_weight=False
        # )
        # print(metrics)

        # Perform edits and check metrics
        editable_model.edit(
            prompts=prompts,
            ground_truth=ground_truth,
            target_new=target_new,
            subject=subject,
            # rephrase_prompts=rephrase_prompt,
            locality_inputs=locality_inputs,
            keep_original_weight=False
        )

        # Check results
        # print(metrics)



    # Save parameters
    # torch.save(editable_model.state_dict(), '/scratch/sux7mp/out/checkpoint.pth')

    # Check if model editing actually edited the model
    # def generate_fingerprint(m):
    #     hash_obj = hashlib.sha256()
    #     [hash_obj.update(p.cpu().detach().numpy().tobytes()) for p in m.parameters()]
    #     return hash_obj.hexdigest()
    # print(
    #     f"Are models identical? {generate_fingerprint(editable_model) == generate_fingerprint(model)}")

    # Sparsify editable model
    pruning_and_validation = LLMPruningAndValidation(args, editable_model.model)

    # Prune
    if config.compress and config.method == 'prune':
        pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
        pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

    # Quant
    if config.compress and config.method == 'quant':
        pruning_and_validation.quantization()

    # metrics_data = editable_model.evaluate(
    #         model=editable_model,
    #         prompts=prompts,
    #         ground_truth=ground_truth,
    #         target_new=target_new,
    #         subject=subject,
    #         # rephrase_prompts=rephrase_prompt,
    #         locality_inputs=locality_inputs,
    #         keep_original_weight=False
    #     )
    # average_rewrite_acc, average_locality = evals.calculate_avg(metrics_data)
    

    locality_score = evals.F1_locality(model, locality_inputs, config, max_length=512)
    success_score = evals.calculate_edit_accuracy(model, prompts, ground_truth, config, max_length=512)
    writer.add_scalar("Rewrite accuracy", success_score, 1)
    writer.add_scalar("Locality", locality_score, 1)
    print(f"Locality: {locality_score}")
    print(f"Success: {success_score}")
    # quit()

    # Validate
    ppl_test = pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
    writer.add_scalar("PPL", ppl_test, 1)

    # Save
    if config.save_ckpt:
        save_ckpt_meta.save(editable_model, config, '/scratch/sux7mp/saved_models/')

if __name__ == '__main__':
    main()
