"""Typer CLI entry point for the Gemma3 fine-tuning pipeline."""

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
    """Set up cache directory and log in to Hugging Face."""
    cfg = PipelineConfig()
    setup_cache(cfg.model.mlx_path)
    login_hf()


@app.command()
def convert() -> None:
    """Convert a Hugging Face model to MLX format."""
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
    """Run the full training pipeline: convert, load, LoRA, data, train."""
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
    adapter_file = setup_adapter(cfg.lora.adapter_path, lora_config)
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
    """Fuse LoRA adapters into the base model."""
    cfg = PipelineConfig()
    fuse_model(
        model_path=cfg.model.mlx_path,
        adapter_path=cfg.lora.adapter_path,
        save_path=cfg.fuse.fuse_path,
    )


@app.command()
def upload() -> None:
    """Upload the fused model to Hugging Face Hub."""
    cfg = PipelineConfig()
    upload_model(
        folder_path=cfg.fuse.fuse_path,
        repo_id=cfg.upload.repo_id,
    )


@app.command()
def test() -> None:
    """Load the fused model and run a test inference."""
    cfg = PipelineConfig()
    model, tokenizer = load_model(cfg.fuse.fuse_path)
    response = run_inference(model, tokenizer, prompt="Hello, world!")
    typer.echo(response)


if __name__ == "__main__":
    app()
