import pytest
from pydantic import ValidationError

from config import (
    DataConfig,
    FuseConfig,
    LoraConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
    UploadConfig,
)


class TestModelConfigDefaults:
    def test_model_name(self):
        cfg = ModelConfig()
        assert cfg.model_name == "google/functiongemma-270m-it"

    def test_quantize(self):
        cfg = ModelConfig()
        assert cfg.quantize is True

    def test_dtype(self):
        cfg = ModelConfig()
        assert cfg.dtype == "float16"

    def test_q_bits(self):
        cfg = ModelConfig()
        assert cfg.q_bits == 4

    def test_mlx_path(self):
        cfg = ModelConfig()
        assert cfg.mlx_path == "./gemma3f-mlx"


class TestLoraConfigDefaults:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.num_layers == 4
        assert cfg.rank == 4
        assert cfg.scale == 20.0
        assert cfg.dropout == 0.0
        assert cfg.adapter_path == "adaptersf_gemma3"


class TestTrainingConfigDefaults:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 1
        assert cfg.iters == 200
        assert cfg.steps_per_eval == 50
        assert cfg.grad_checkpoint is True
        assert cfg.learning_rate == 1e-5


class TestDataConfigDefaults:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.hf_dataset_path == "google/mobile-actions"
        assert cfg.train_split == "train[:70%]"
        assert cfg.valid_split == "train[-10%:]"
        assert cfg.prompt_feature == "prompt"
        assert cfg.completion_feature == "response"


class TestFuseConfigDefaults:
    def test_defaults(self):
        cfg = FuseConfig()
        assert cfg.fuse_path == "./new_functiongemma3"


class TestUploadConfigDefaults:
    def test_defaults(self):
        cfg = UploadConfig()
        assert cfg.repo_id == "deimagjas/functiongemma-3-270m-it-sft"


class TestEnvVarOverrides:
    def test_model_name_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_MODEL_NAME", "google/gemma-3-1b-it")
        cfg = ModelConfig()
        assert cfg.model_name == "google/gemma-3-1b-it"

    def test_q_bits_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_Q_BITS", "8")
        cfg = ModelConfig()
        assert cfg.q_bits == 8

    def test_training_iters_override(self, monkeypatch):
        monkeypatch.setenv("TRAINING_ITERS", "500")
        cfg = TrainingConfig()
        assert cfg.iters == 500

    def test_learning_rate_override(self, monkeypatch):
        monkeypatch.setenv("TRAINING_LEARNING_RATE", "0.001")
        cfg = TrainingConfig()
        assert cfg.learning_rate == 0.001

    def test_data_path_override(self, monkeypatch):
        monkeypatch.setenv("DATA_HF_DATASET_PATH", "custom/dataset")
        cfg = DataConfig()
        assert cfg.hf_dataset_path == "custom/dataset"

    def test_lora_rank_override(self, monkeypatch):
        monkeypatch.setenv("LORA_RANK", "16")
        cfg = LoraConfig()
        assert cfg.rank == 16


class TestValidation:
    def test_q_bits_must_be_positive(self):
        with pytest.raises(ValidationError):
            ModelConfig(q_bits=0)

    def test_q_bits_rejects_negative(self):
        with pytest.raises(ValidationError):
            ModelConfig(q_bits=-1)

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_iters_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingConfig(iters=-10)

    def test_learning_rate_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)

    def test_dropout_max_bound(self):
        with pytest.raises(ValidationError):
            LoraConfig(dropout=1.5)

    def test_dropout_min_bound(self):
        with pytest.raises(ValidationError):
            LoraConfig(dropout=-0.1)

    def test_lora_rank_must_be_positive(self):
        with pytest.raises(ValidationError):
            LoraConfig(rank=0)

    def test_num_layers_must_be_positive(self):
        with pytest.raises(ValidationError):
            LoraConfig(num_layers=0)


class TestPipelineConfig:
    def test_contains_all_sub_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.fuse, FuseConfig)
        assert isinstance(cfg.upload, UploadConfig)

    def test_sub_config_defaults_preserved(self):
        cfg = PipelineConfig()
        assert cfg.model.model_name == "google/functiongemma-270m-it"
        assert cfg.lora.rank == 4
        assert cfg.training.iters == 200
        assert cfg.data.hf_dataset_path == "google/mobile-actions"
        assert cfg.fuse.fuse_path == "./new_functiongemma3"
        assert cfg.upload.repo_id == "deimagjas/functiongemma-3-270m-it-sft"

    def test_sub_config_override(self):
        cfg = PipelineConfig(
            model=ModelConfig(model_name="custom/model"),
            training=TrainingConfig(iters=500),
        )
        assert cfg.model.model_name == "custom/model"
        assert cfg.training.iters == 500
