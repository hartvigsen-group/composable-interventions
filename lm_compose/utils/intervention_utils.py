import torch


def get_dtype(dtype_str):
    """Dynamically get the torch dtype based on the config"""
    dtype_mapping = {
        "torch.float": torch.float,
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float64": torch.float64,  # Adding more possible dtypes
        "torch.half": torch.half,
        "torch.double": torch.double,
        "awq": torch.float16,
        "gptq": torch.float16,
        "wanda": torch.bfloat16,
        "sparsegpt": torch.bfloat16,
        "ft": torch.bfloat16,
        "memit": torch.bfloat16,
        "lora": torch.float,
        "rmu": torch.bfloat16,
        "ga": torch.bfloat16,
    }

    if dtype_str not in dtype_mapping:
        raise ValueError(f"Invalid dtype specified in config: {dtype_str}")

    return dtype_mapping[dtype_str]
