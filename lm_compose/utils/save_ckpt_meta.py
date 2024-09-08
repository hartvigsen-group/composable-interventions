import datetime
import os

import torch
from omegaconf import OmegaConf


def get_timestamp():
    # Generates a timestamp string
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save(model, config, timestamp, checkpoint_dir):
    # Ensure the directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{timestamp}.pth")
    torch.save(model.state_dict(), checkpoint_path)

    # Save the metadata (hyperparameters)
    meta_path = os.path.join(checkpoint_dir, f"meta_{timestamp}.yaml")
    OmegaConf.save(config, meta_path)
    # with open(meta_path, 'w') as meta_file:
    #     yaml.dump(config, meta_file, default_flow_style=False)

    print(f"Model checkpoint saved to {checkpoint_path}")
    print(f"Metadata saved to {meta_path}")
