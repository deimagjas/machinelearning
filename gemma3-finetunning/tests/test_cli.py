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
        # Given
        expected_cache_dir = cfg.model.cache_dir

        # When
        result = runner.invoke(app, ["login"])

        # Then
        assert result.exit_code == 0
        mock_cache.assert_called_once_with(expected_cache_dir)
        mock_login.assert_called_once()


class TestConvertCommand:
    @patch("cli.convert_model")
    def test_calls_convert_model_with_config(self, mock_convert, cfg):
        # Given
        m = cfg.model

        # When
        result = runner.invoke(app, ["convert"])

        # Then
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
        # Given
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

        # When
        result = runner.invoke(app, ["train"])

        # Then
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
        # Given
        expected_model_path = cfg.model.mlx_path
        expected_adapter_path = cfg.lora.adapter_path
        expected_save_path = cfg.fuse.fuse_path

        # When
        result = runner.invoke(app, ["fuse"])

        # Then
        assert result.exit_code == 0
        mock_fuse.assert_called_once_with(
            model_path=expected_model_path,
            adapter_path=expected_adapter_path,
            save_path=expected_save_path,
        )


class TestUploadCommand:
    @patch("cli.upload_model")
    def test_calls_upload_model_with_config(self, mock_upload, cfg):
        # Given
        expected_folder_path = cfg.fuse.fuse_path
        expected_repo_id = cfg.upload.repo_id

        # When
        result = runner.invoke(app, ["upload"])

        # Then
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
        # Given
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_inference.return_value = "Generated response"

        # When
        result = runner.invoke(app, ["test"])

        # Then
        assert result.exit_code == 0
        mock_load.assert_called_once_with(cfg.fuse.fuse_path)
        mock_inference.assert_called_once_with(
            mock_model, mock_tokenizer, prompt="Hello, world!"
        )

    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_echoes_inference_response(self, mock_load, mock_inference):
        # Given
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.return_value = "Test output"

        # When
        result = runner.invoke(app, ["test"])

        # Then
        assert result.exit_code == 0
        assert "Test output" in result.output


class TestInferenceCommand:
    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_loads_base_model_and_runs_inference(self, mock_load, mock_inference, cfg):
        # Given
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)
        mock_inference.return_value = "Base model response"
        prompt = "Explain gravity"

        # When
        result = runner.invoke(app, ["inference", prompt])

        # Then
        assert result.exit_code == 0
        mock_load.assert_called_once_with(cfg.model.mlx_path)
        mock_inference.assert_called_once_with(mock_model, mock_tokenizer, prompt)

    @patch("cli.run_inference")
    @patch("cli.load_model")
    def test_echoes_inference_response(self, mock_load, mock_inference):
        # Given
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_inference.return_value = "Base output"

        # When
        result = runner.invoke(app, ["inference", "Hello"])

        # Then
        assert result.exit_code == 0
        assert "Base output" in result.output
