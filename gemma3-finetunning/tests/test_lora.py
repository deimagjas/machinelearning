import json
import os
from unittest.mock import MagicMock, patch

from pipeline.lora import apply_lora, setup_adapter


class TestSetupAdapter:
    def test_creates_directory_and_config(self, tmp_path):
        adapter_path = str(tmp_path / "adapters")
        lora_params = {"rank": 4, "scale": 20.0, "dropout": 0.0}

        result = setup_adapter(adapter_path, num_layers=4, lora_params=lora_params)

        assert os.path.isdir(adapter_path)
        assert result == os.path.join(adapter_path, "adapters.safetensors")
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            saved = json.load(f)
        assert saved == {"num_layers": 4, "lora_parameters": lora_params}

    def test_overwrites_existing_config(self, tmp_path):
        adapter_path = str(tmp_path / "adapters")
        os.makedirs(adapter_path)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump({"old": True}, f)

        setup_adapter(adapter_path, num_layers=8, lora_params={"rank": 8})

        with open(config_path) as f:
            saved = json.load(f)
        assert saved == {"num_layers": 8, "lora_parameters": {"rank": 8}}


class TestApplyLora:
    @patch("pipeline.lora.linear_to_lora_layers")
    @patch("pipeline.lora.tree_flatten")
    def test_freezes_and_applies_lora(self, mock_flatten, mock_lora_layers):
        model = MagicMock()
        mock_param = MagicMock()
        mock_param.size = 1000
        mock_flatten.return_value = [("w", mock_param)]

        lora_params = {"rank": 4, "scale": 20.0, "dropout": 0.0}
        result = apply_lora(model, 4, lora_params)

        model.freeze.assert_called_once()
        mock_lora_layers.assert_called_once_with(model, 4, lora_params)
        model.train.assert_called_once()
        assert result is model
