"""Tests for the Typer CLI entry point."""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cli import app
from config import PipelineConfig

runner = CliRunner()


@pytest.fixture()
def cfg():
    return PipelineConfig()


class TestLoginCommand:
    @patch("cli.login_hf")
    @patch("cli.setup_cache")
    def test_calls_setup_cache_and_login(self, mock_cache, mock_login, cfg):
        # Given — mocked HuggingFace login and cache setup functions, with the expected cache directory from config
        expected_cache_dir = cfg.model.cache_dir

        # When — the "login" CLI command is invoked
        result = runner.invoke(app, ["login"])

        # Then — the command exits successfully, sets up the model cache directory, and triggers HF login
        assert result.exit_code == 0
        mock_cache.assert_called_once_with(expected_cache_dir)
        mock_login.assert_called_once()


class TestConvertCommand:
    @patch("cli.convert_model")
    def test_calls_convert_model_with_config(self, mock_convert, cfg):
        # Given — a mocked convert_model function and the default model configuration
        m = cfg.model

        # When — the "convert" CLI command is invoked
        result = runner.invoke(app, ["convert"])

        # Then — convert_model is called with all model config values (name, path, quantize, dtype, q_bits)
        assert result.exit_code == 0
        mock_convert.assert_called_once_with(
            model_name=m.model_name,
            mlx_path=m.mlx_path,
            quantize=m.quantize,
            dtype=m.dtype,
            q_bits=m.q_bits,
        )


class TestTrainCommand:
    @patch("cli.run_training")
    @patch("cli.load_training_data")
    @patch("cli.apply_lora")
    @patch("cli.setup_adapter")
    @patch("cli.load_model")
    @patch("cli.convert_model")
    def test_full_training_pipeline(
        self,
        mock_convert,
        mock_load_model,
        mock_setup_adapter,
        mock_apply_lora,
        mock_load_data,
        mock_train,
        cfg,
    ):
        # Given — all pipeline steps (convert, load, adapter setup, LoRA, data, train) are mocked with realistic return values
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_setup_adapter.return_value = "adapter.safetensors"
        mock_load_data.return_value = (["train"], ["val"])

        lora_config = {
            "rank": cfg.lora.rank,
            "scale": cfg.lora.scale,
            "dropout": cfg.lora.dropout,
        }
        training_args = {
            "batch_size": cfg.training.batch_size,
            "iters": cfg.training.iters,
            "steps_per_eval": cfg.training.steps_per_eval,
            "grad_checkpoint": cfg.training.grad_checkpoint,
        }

        # When — the "train" CLI command is invoked, triggering the full training pipeline
        result = runner.invoke(app, ["train"])

        # Then — the command exits successfully and each pipeline step is called with the correct config values
        assert result.exit_code == 0

        mock_convert.assert_called_once_with(
            model_name=cfg.model.model_name,
            mlx_path=cfg.model.mlx_path,
            quantize=cfg.model.quantize,
            dtype=cfg.model.dtype,
            q_bits=cfg.model.q_bits,
        )

        mock_load_model.assert_called_once_with(cfg.model.mlx_path)

        mock_setup_adapter.assert_called_once_with(
            cfg.lora.adapter_path, cfg.lora.num_layers, lora_config
        )
        mock_apply_lora.assert_called_once_with(
            mock_model, cfg.lora.num_layers, lora_config
        )

        mock_load_data.assert_called_once_with(
            mock_tokenizer,
            hf_dataset_path=cfg.data.hf_dataset_path,
            train_split=cfg.data.train_split,
            valid_split=cfg.data.valid_split,
            prompt_feature=cfg.data.prompt_feature,
            completion_feature=cfg.data.completion_feature,
        )

        mock_train.assert_called_once_with(
            mock_model,
            ["train"],
            ["val"],
            training_args_dict=training_args,
            adapter_file="adapter.safetensors",
            learning_rate=cfg.training.learning_rate,
        )


class TestFuseCommand:
    @patch("cli.fuse_model")
    def test_calls_fuse_model_with_config(self, mock_fuse, cfg):
        # Given — a mocked fuse_model function and config paths for model, adapter, and output
        expected_model_path = cfg.model.mlx_path
        expected_adapter_path = cfg.lora.adapter_path
        expected_save_path = cfg.fuse.fuse_path

        # When — the "fuse" CLI command is invoked
        result = runner.invoke(app, ["fuse"])

        # Then — fuse_model is called with the MLX model path, adapter path, and fuse output path from config
        assert result.exit_code == 0
        mock_fuse.assert_called_once_with(
            model_path=expected_model_path,
            adapter_path=expected_adapter_path,
            save_path=expected_save_path,
        )


class TestUploadCommand:
    @patch("cli.upload_model")
    def test_calls_upload_model_with_config(self, mock_upload, cfg):
        # Given — a mocked upload_model function and the expected folder path and repo ID from config
        expected_folder_path = cfg.fuse.fuse_path
        expected_repo_id = cfg.upload.repo_id

        # When — the "upload" CLI command is invoked
        result = runner.invoke(app, ["upload"])

        # Then — upload_model is called with the fused model folder and the target HuggingFace repo ID
        assert result.exit_code == 0
        mock_upload.assert_called_once_with(
            folder_path=expected_folder_path,
            repo_id=expected_repo_id,
        )


class TestTestCommand:
    @patch("cli.typer.echo")
    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_loads_fused_model_and_runs_inference(
        self, mock_load, mock_inference, mock_echo, cfg
    ):
        # Given — mocked load_model returning a fake model/tokenizer pair, and run_inference returning canned text
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_inference.return_value = "Generated response"

        # When — the "test" CLI command is invoked to run inference on the fused model
        result = runner.invoke(app, ["test"])

        # Then — the fused model is loaded from config path and inference runs with the default "Hello, world!" prompt
        assert result.exit_code == 0
        mock_load.assert_called_once_with(cfg.fuse.fuse_path)
        mock_inference.assert_called_once_with(
            mock_model, mock_tokenizer, prompt="Hello, world!"
        )

    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_echoes_inference_response(self, mock_load, mock_inference):
        # Given — mocked model loading and inference returning "Test output"
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.return_value = "Test output"

        # When — the "test" CLI command is invoked
        result = runner.invoke(app, ["test"])

        # Then — the inference response text appears in the CLI output
        assert result.exit_code == 0
        assert "Test output" in result.output


class TestInferenceCommand:
    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_loads_base_model_and_runs_inference(self, mock_load, mock_inference, cfg):
        # Given — mocked model loading and inference, with a user-provided prompt
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_inference.return_value = "Base model response"
        prompt = "Explain gravity"

        # When — the "inference" CLI command is invoked with the prompt argument
        result = runner.invoke(app, ["inference", prompt])

        # Then — the base model (not fused) is loaded from mlx_path and inference runs with the given prompt
        assert result.exit_code == 0
        mock_load.assert_called_once_with(cfg.model.mlx_path)
        mock_inference.assert_called_once_with(mock_model, mock_tokenizer, prompt)

    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_echoes_inference_response(self, mock_load, mock_inference):
        # Given — mocked model loading and inference returning "Base output"
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.return_value = "Base output"

        # When — the "inference" CLI command is invoked with a simple prompt
        result = runner.invoke(app, ["inference", "Hello"])

        # Then — the inference response text appears in the CLI output
        assert result.exit_code == 0
        assert "Base output" in result.output
