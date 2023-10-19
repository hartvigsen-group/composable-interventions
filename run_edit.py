import torch
from easyeditor import MEMITHyperParams
from easyeditor import BaseEditor, ModelEditWrapper
import os
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer
import hashlib

# Define parameters

hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama.yaml')

prompts = ['Who was the designer of Lahti Town Hall?',
           'What role does Denny Herzig play in football?',
           'What city did Marl Young live when he died?']
ground_truth = ['Eliel Saarinen', 'defender', 'Los Angeles']
target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['Lahti Town Hall', 'Denny Herzig', 'Marl Young']

# init some model

# model = GPTNeoXForCausalLM.from_pretrained(
#   "EleutherAI/pythia-70m-deduped",
#   revision="step3000")

model = AutoModelForCausalLM.from_pretrained(
            'decapoda-research/llama-7b-hf',
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            device_map="auto"
        )

# Wrap the original model to make it editable

editable_model = ModelEditWrapper(model, hparams)

# Editable model behaves likes a normal pytorch model

# torch.save(editable_model.state_dict(), 'checkpoint.pth')

# Load the state_dict
# state_dict = torch.load('checkpoint.pth')

# Update the model's state_dict
# editable_model.load_state_dict(state_dict)

# Do some model editing

metrics, edited_model = editable_model.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)

# Check results
print(metrics)
print(type(edited_model))

# Check if model editing actually edited the model

def generate_fingerprint(m): return hashlib.sha256(b''.join(
    p.cpu().detach().numpy().tobytes() for p in m.parameters())).hexdigest()

print(
    f"Are models identical? {generate_fingerprint(model) == generate_fingerprint(edited_model)}")