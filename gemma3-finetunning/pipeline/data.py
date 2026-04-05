import types

from mlx_lm.tuner.datasets import load_custom_hf_dataset


def load_training_data(
    tokenizer,
    hf_dataset_path: str,
    train_split: str,
    valid_split: str,
    prompt_feature: str,
    completion_feature: str,
) -> tuple:
    args = types.SimpleNamespace(
        hf_dataset={
            "path": hf_dataset_path,
            "train_split": train_split,
            "valid_split": valid_split,
            "prompt_feature": prompt_feature,
            "completion_feature": completion_feature,
            "config": {},
        },
        mask_prompt=False,
        train=True,
        test=False,
    )
    train_set, val_set, _ = load_custom_hf_dataset(args=args, tokenizer=tokenizer)
    return train_set, val_set
