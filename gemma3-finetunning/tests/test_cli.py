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
        result = runner.invoke(app, ["login"])

        assert result.exit_code == 0
        mock_cache.assert_called_once_with(cfg.model.mlx_path)
        mock_login.assert_called_once()


class TestConvertCommand:
    @patch("cli.convert_model")
    def test_calls_convert_model_with_config(self, mock_convert, cfg):
        result = runner.invoke(app, ["convert"])

        assert result.exit_code == 0
        mock_convert.assert_called_once_with(
            model_name=cfg.model.model_name,
            mlx_path=cfg.model.mlx_path,
            quantize=cfg.model.quantize,
            dtype=cfg.model.dtype,
            q_bits=cfg.model.q_bits,
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
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_model, mock_tokenizer)
        mock_setup_adapter.return_value = "adapter.safetensors"
        mock_load_data.return_value = (["train"], ["val"])

        result = runner.invoke(app, ["train"])

        assert result.exit_code == 0

        mock_convert.assert_called_once_with(
            model_name=cfg.model.model_name,
            mlx_path=cfg.model.mlx_path,
            quantize=cfg.model.quantize,
            dtype=cfg.model.dtype,
            q_bits=cfg.model.q_bits,
        )

        mock_load_model.assert_called_once_with(cfg.model.mlx_path)

        lora_config = {
            "rank": cfg.lora.rank,
            "scale": cfg.lora.scale,
            "dropout": cfg.lora.dropout,
        }
        mock_setup_adapter.assert_called_once_with(
            cfg.lora.adapter_path, lora_config
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

        training_args = {
            "batch_size": cfg.training.batch_size,
            "iters": cfg.training.iters,
            "steps_per_eval": cfg.training.steps_per_eval,
            "grad_checkpoint": cfg.training.grad_checkpoint,
        }
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
        result = runner.invoke(app, ["fuse"])

        assert result.exit_code == 0
        mock_fuse.assert_called_once_with(
            model_path=cfg.model.mlx_path,
            adapter_path=cfg.lora.adapter_path,
            save_path=cfg.fuse.fuse_path,
        )


class TestUploadCommand:
    @patch("cli.upload_model")
    def test_calls_upload_model_with_config(self, mock_upload, cfg):
        result = runner.invoke(app, ["upload"])

        assert result.exit_code == 0
        mock_upload.assert_called_once_with(
            folder_path=cfg.fuse.fuse_path,
            repo_id=cfg.upload.repo_id,
        )


class TestTestCommand:
    @patch("cli.typer.echo")
    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_loads_fused_model_and_runs_inference(
        self, mock_load, mock_inference, mock_echo, cfg
    ):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_inference.return_value = "Generated response"

        result = runner.invoke(app, ["test"])

        assert result.exit_code == 0
        mock_load.assert_called_once_with(cfg.fuse.fuse_path)
        mock_inference.assert_called_once_with(
            mock_model, mock_tokenizer, prompt="Hello, world!"
        )

    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_echoes_inference_response(self, mock_load, mock_inference):
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.return_value = "Test output"

        result = runner.invoke(app, ["test"])

        assert result.exit_code == 0
        assert "Test output" in result.output
