import json
import os

from mlx.utils import tree_flatten
from mlx_lm.tuner import linear_to_lora_layers


def setup_adapter(adapter_path: str, lora_config_dict: dict) -> str:
    os.makedirs(adapter_path, exist_ok=True)
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_config_path, "w") as f:
        json.dump(lora_config_dict, f, indent=4)
    return adapter_config_path


def apply_lora(model, num_layers: int, lora_params: dict):
    model.freeze()
    linear_to_lora_layers(model, num_layers, lora_params)
    num_train_params = sum(
        v.size for _, v in tree_flatten(model.trainable_parameters())
    )
    print(f"Number of trainable parameters: {num_train_params}")
    model.train()
    return model
