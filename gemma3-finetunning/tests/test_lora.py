import json
import os
from unittest.mock import MagicMock, patch

from pipeline.lora import apply_lora, setup_adapter


class TestSetupAdapter:
    def test_creates_directory_and_config(self, tmp_path):
        # Given — a non-existent adapter directory and LoRA parameters
        adapter_path = str(tmp_path / "adapters")
        lora_params = {"rank": 4, "scale": 20.0, "dropout": 0.0}

        # When — setup_adapter is called to initialize the adapter directory
        result = setup_adapter(adapter_path, num_layers=4, lora_params=lora_params)

        # Then — the directory is created, the adapter file path is returned, and config JSON contains num_layers and lora_parameters
        assert os.path.isdir(adapter_path)
        assert result == os.path.join(adapter_path, "adapters.safetensors")
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            saved = json.load(f)
        assert saved == {"num_layers": 4, "lora_parameters": lora_params}

    def test_overwrites_existing_config(self, tmp_path):
        # Given — an adapter directory that already contains an old config file
        adapter_path = str(tmp_path / "adapters")
        os.makedirs(adapter_path)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump({"old": True}, f)

        # When — setup_adapter is called with new parameters
        setup_adapter(adapter_path, num_layers=8, lora_params={"rank": 8})

        # Then — the old config is replaced with the new num_layers and lora_parameters
        with open(config_path) as f:
            saved = json.load(f)
        assert saved == {"num_layers": 8, "lora_parameters": {"rank": 8}}


class TestApplyLora:
    @patch("pipeline.lora.linear_to_lora_layers")
    @patch("pipeline.lora.tree_flatten")
    def test_freezes_and_applies_lora(self, mock_flatten, mock_lora_layers):
        # Given — a mock model with one parameter and mocked LoRA utilities
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.size = 1000
        mock_flatten.return_value = [("w", mock_param)]

        # When — apply_lora is called with 4 layers and LoRA parameters
        lora_params = {"rank": 4, "scale": 20.0, "dropout": 0.0}
        result = apply_lora(model, 4, lora_params)

        # Then — the model is frozen, LoRA layers are applied, training mode is enabled, and the same model is returned
        model.freeze.assert_called_once()
        mock_lora_layers.assert_called_once_with(model, 4, lora_params)
        model.train.assert_called_once()
        assert result is model
