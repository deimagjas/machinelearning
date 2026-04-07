from unittest.mock import MagicMock, patch

from pipeline.data import load_training_data


class TestLoadTrainingData:
    @patch("pipeline.data.load_custom_hf_dataset")
    def test_returns_train_and_val_sets(self, mock_load):
        # Given — a mocked HuggingFace dataset loader returning predefined train and validation splits
        train_data = [{"text": "train"}]
        val_data = [{"text": "val"}]
        mock_load.return_value = (train_data, val_data, [])
        tokenizer = MagicMock()

        # When — load_training_data is called with dataset path, splits, and feature names
        train_set, val_set = load_training_data(
            tokenizer=tokenizer,
            hf_dataset_path="google/mobile-actions",
            train_split="train[:70%]",
            valid_split="train[-10%:]",
            prompt_feature="prompt",
            completion_feature="response",
        )

        # Then — the correct train/val data is returned and the loader receives properly structured arguments
        assert train_set == train_data
        assert val_set == val_data
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args
        args_ns = call_kwargs.kwargs["args"]
        assert args_ns.hf_dataset["path"] == "google/mobile-actions"
        assert args_ns.hf_dataset["prompt_feature"] == "prompt"
        assert args_ns.hf_dataset["completion_feature"] == "response"
        assert args_ns.mask_prompt is False
        assert args_ns.train is True
        assert args_ns.test is False
