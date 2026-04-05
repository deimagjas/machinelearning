from typing import Dict, List, Tuple, Union

import mlx.optimizers as optim
from mlx_lm.tuner import TrainingArgs, train
from mlx_lm.tuner.datasets import CacheDataset


class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))


def run_training(
    model,
    train_set: list,
    val_set: list,
    training_args_dict: dict,
    adapter_file: str,
    learning_rate: float,
) -> Metrics:
    train_dataset = CacheDataset(train_set)
    val_dataset = CacheDataset(val_set)

    training_args = TrainingArgs(
        adapter_file=adapter_file,
        **training_args_dict,
    )

    metrics = Metrics()
    optimizer = optim.Adam(learning_rate=learning_rate)

    train(
        model,
        optimizer,
        train_dataset,
        val_dataset,
        args=training_args,
        training_callback=metrics,
    )
    return metrics
