import json
import os
import sys
import subprocess

w_bit_levels=[2, 4, 8]
intervention_orders = [
    ("[unlearn,edit,compress]", "rmu-to-lora-to-AWQ{BIT}bit"),
    ("[unlearn,compress,edit]", "rmu-to-AWQ{BIT}bit-to-lora"),
    ("[edit,unlearn,compress]", "lora-to-rmu-to-AWQ{BIT}bit"),
    ("[edit,compress,unlearn]", "lora-to-AWQ{BIT}bit-to-rmu"),
    ("[compress,unlearn,edit]", "AWQ{BIT}bit-to-rmu-to-lora"),
    ("[compress,edit,unlearn]", "AWQ{BIT}bit-to-lora-to-rmu")
]

configs = []
for intervention_order, tag in intervention_orders:
    for w_bit in w_bit_levels:
        tag = tag.format(BIT=w_bit)
        configs.append(f"edit=lora compression=awq unlearn=rmu interventions={intervention_order} wbits={w_bit} tag={tag}")

for experiment_config in configs:
    # os.system(f"sbatch run_exp.sh {experiment_config}")
    print(f"sbatch run_exp.sh {experiment_config}")