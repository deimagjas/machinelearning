"""Typer CLI entry point for the Gemma3 fine-tuning pipeline.

This module provides a command-line interface for fine-tuning Google's Gemma 3
model on Apple Silicon using MLX. Each pipeline stage is exposed as a standalone
Typer command that can be invoked individually or chained together.

Configuration
-------------
All settings are managed through ``PipelineConfig`` (Pydantic-settings).
Values can be overridden via environment variables using section prefixes:

- ``MODEL_*``     – model name, MLX path, quantization options
- ``LORA_*``      – LoRA rank, scale, dropout, adapter path
- ``TRAINING_*``  – batch size, iterations, learning rate
- ``DATA_*``      – dataset path, splits, feature names
- ``FUSE_*``      – fused model output path
- ``UPLOAD_*``    – Hugging Face repo ID for upload

Example usage
-------------
Via Make targets::

    make login          # authenticate with Hugging Face
    make convert        # convert HF model to MLX format
    make train          # run full training pipeline
    make fuse           # merge LoRA adapters into base model
    make upload         # push fused model to HF Hub
    make test           # run test inference on fused model
    make inference PROMPT="Explain gravity"  # run inference on base model

Via Python directly::

    python cli.py login
    python cli.py convert
    python cli.py train
    python cli.py inference "Explain gravity"

Pipeline flow
-------------
.. code-block:: mermaid

    graph LR
        login["login"] --> convert["convert"]
        convert --> train["train"]
        train --> fuse["fuse"]
        fuse --> upload["upload"]
        fuse --> test["test"]
        convert --> inference["inference"]
"""

import typer

from config import PipelineConfig
from pipeline.data import load_training_data
from pipeline.environment import login_hf, setup_cache
from pipeline.fuse import fuse_model
from pipeline.inference import run_inference
from pipeline.lora import apply_lora, setup_adapter
from pipeline.model import convert_model, load_model
from pipeline.train import run_training
from pipeline.upload import upload_model

app = typer.Typer()


@app.command()
def login() -> None:
    """Set up the local cache directory and authenticate with Hugging Face.

    Creates the cache directory specified in MODEL_CACHE_DIR (if it does not
    exist) and performs an interactive Hugging Face Hub login so that
    subsequent commands can download gated models and push artifacts.
    """
    cfg = PipelineConfig()
    setup_cache(cfg.model.cache_dir)
    login_hf()


@app.command()
def convert() -> None:
    """Convert a Hugging Face model to MLX format.

    Downloads the model specified by MODEL_MODEL_NAME from the Hugging Face Hub
    and converts it into the MLX weight format, optionally applying
    quantization (controlled by MODEL_QUANTIZE, MODEL_DTYPE, MODEL_Q_BITS).
    The converted weights are saved to MODEL_MLX_PATH.
    """
    cfg = PipelineConfig()
    m = cfg.model
    convert_model(
        model_name=m.model_name,
        mlx_path=m.mlx_path,
        quantize=m.quantize,
        dtype=m.dtype,
        q_bits=m.q_bits,
    )


@app.command()
def train() -> None:
    """Run the full training pipeline: convert, load, LoRA, data, train.

    Orchestrates every step needed to fine-tune the model:
    1. Convert the HF model to MLX format (skipped if already converted).
    2. Load the converted model and tokenizer.
    3. Initialise LoRA adapters according to LORA_* settings.
    4. Load and tokenize the training/validation splits from DATA_HF_DATASET_PATH.
    5. Run the training loop with TRAINING_* hyperparameters.
    """
    cfg = PipelineConfig()

    convert_model(
        model_name=cfg.model.model_name,
        mlx_path=cfg.model.mlx_path,
        quantize=cfg.model.quantize,
        dtype=cfg.model.dtype,
        q_bits=cfg.model.q_bits,
    )

    model, tokenizer = load_model(cfg.model.mlx_path)

    lora_config = {
        "rank": cfg.lora.rank,
        "scale": cfg.lora.scale,
        "dropout": cfg.lora.dropout,
    }
    adapter_file = setup_adapter(cfg.lora.adapter_path, cfg.lora.num_layers, lora_config)
    apply_lora(model, cfg.lora.num_layers, lora_config)

    train_set, val_set = load_training_data(
        tokenizer,
        hf_dataset_path=cfg.data.hf_dataset_path,
        train_split=cfg.data.train_split,
        valid_split=cfg.data.valid_split,
        prompt_feature=cfg.data.prompt_feature,
        completion_feature=cfg.data.completion_feature,
    )

    training_args = {
        "batch_size": cfg.training.batch_size,
        "iters": cfg.training.iters,
        "steps_per_eval": cfg.training.steps_per_eval,
        "grad_checkpoint": cfg.training.grad_checkpoint,
    }
    run_training(
        model,
        train_set,
        val_set,
        training_args_dict=training_args,
        adapter_file=adapter_file,
        learning_rate=cfg.training.learning_rate,
    )


@app.command()
def fuse() -> None:
    """Fuse LoRA adapters into the base model.

    Merges the trained LoRA adapter weights (from LORA_ADAPTER_PATH) back into
    the base MLX model (MODEL_MLX_PATH) and writes the resulting full-precision
    model to FUSE_FUSE_PATH.
    """
    cfg = PipelineConfig()
    fuse_model(
        model_path=cfg.model.mlx_path,
        adapter_path=cfg.lora.adapter_path,
        save_path=cfg.fuse.fuse_path,
    )


@app.command()
def upload() -> None:
    """Upload the fused model to Hugging Face Hub.

    Pushes the fused model directory (FUSE_FUSE_PATH) to the Hugging Face Hub
    repository specified by UPLOAD_REPO_ID. Requires prior authentication
    via the ``login`` command.
    """
    cfg = PipelineConfig()
    upload_model(
        folder_path=cfg.fuse.fuse_path,
        repo_id=cfg.upload.repo_id,
    )


@app.command()
def test() -> None:
    """Load the fused model and run a test inference.

    Loads the fused model from FUSE_FUSE_PATH and generates a response to the
    hard-coded prompt ``"Hello, world!"`` as a quick smoke-test to verify that
    fusion succeeded.
    """
    cfg = PipelineConfig()
    model, tokenizer = load_model(cfg.fuse.fuse_path)
    response = run_inference(model, tokenizer, prompt="Hello, world!")
    typer.echo(response)


@app.command()
def inference(
    prompt: str = typer.Argument(..., help="Prompt text for model inference"),
) -> None:
    """Run inference on the base converted model (before fine-tuning).

    Loads the converted MLX model from MODEL_MLX_PATH and generates a response
    for the given prompt. Useful for establishing a baseline before training.
    """
    cfg = PipelineConfig()
    model, tokenizer = load_model(cfg.model.mlx_path)
    response = run_inference(model, tokenizer, prompt)
    typer.echo(response)


if __name__ == "__main__":
    app()
