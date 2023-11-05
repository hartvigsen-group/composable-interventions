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

# Set up editing parameters
hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama_fast.yaml')

# yaml file for sparsity
class Config:
    def __init__(self, path):
        with open(path, 'r') as stream:
            data = yaml.load(stream, Loader=yaml.SafeLoader)
            self.__dict__.update(data)
args = Config('hparams/efficiency/quant.yaml')

# TODO: add an edit loader
prompts = ['Who was the designer of Lahti Town Hall?',
           'What role does Denny Herzig play in football?',
           'What city did Marl Young live when he died?']
ground_truth = ['Eliel Saarinen', 'defender', 'Los Angeles']
target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young']

# Set up sparsity parameters
# parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default="decapoda-research/llama-7b-hf", help='LLaMA model')
# parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
# parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
# parser.add_argument('--sparsity_ratio', type=float, default=0.7, help='Sparsity level between 0 and 1')
# parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
# parser.add_argument("--prune_method", type=str, default="wanda", choices=["magnitude", "wanda", "sparsegpt", 
#                     "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
# parser.add_argument("--cache_dir", default="llm_weights", type=str )
# parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
# parser.add_argument('--save', type=str, default="out/", help='Path to save results.')
# parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

# parser.add_argument("--eval_zero_shot", action="store_true")
# args = parser.parse_args()

# Init model
model = AutoModelForCausalLM.from_pretrained(
            'decapoda-research/llama-7b-hf',
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )

# Make editable
editable_model = ModelEditWrapper(model, hparams)

# editable_model.edit(
#     prompts=prompts,
#     ground_truth=ground_truth,
#     target_new=target_new,
#     subject=subject,
#     keep_original_weight=False
# )

# Check results
# print(metrics)
# print(type(edited_model))

# Save parameters
# torch.save(editable_model.state_dict(), '/scratch/sux7mp/out/checkpoint.pth')

# Load the state_dict
state_dict = torch.load('/scratch/sux7mp/out/checkpoint.pth')

# Update the model's state_dict
model.load_state_dict(state_dict)

# Check if model editing actually edited the model
# def generate_fingerprint(m):
#     hash_obj = hashlib.sha256()
#     [hash_obj.update(p.cpu().detach().numpy().tobytes()) for p in m.parameters()]
#     return hash_obj.hexdigest()
# print(
#     f"Are models identical? {generate_fingerprint(editable_model) == generate_fingerprint(model)}")

# Sparsify editable model
pruning_and_validation = LLMPruningAndValidation(args, model)

# Prune
# pruning_and_validation.get_Mask()           #Get Mask with (0,1) for weights, the masks will be saved in self.Masks.  Just do it one time, then fixed it. 
# pruning_and_validation.prune()              # Mask out the weights.   Each time when you changed the updated model weights, then you can need to call this function before you do forward. 

# Quant
pruning_and_validation.quantization()

# Validate
pruning_and_validation.validate()           #It is a validation for general performance on common language benchmark such as wikitext.
