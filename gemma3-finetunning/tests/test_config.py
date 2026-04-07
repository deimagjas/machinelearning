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
    _short_name,
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
        assert cfg.mlx_path == "./functiongemma-270m-it-mlx"


class TestLoraConfigDefaults:
    def test_defaults(self):
        cfg = LoraConfig()
        assert cfg.num_layers == 4
        assert cfg.rank == 4
        assert cfg.scale == 20.0
        assert cfg.dropout == 0.0
        assert cfg.adapter_path == ""


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
        assert cfg.fuse_path == ""


class TestUploadConfigDefaults:
    def test_defaults(self):
        cfg = UploadConfig()
        assert cfg.repo_id == ""


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
        assert cfg.fuse.fuse_path == "./functiongemma-270m-it-new"
        assert cfg.upload.repo_id == "deimagjas/functiongemma-270m-it-mlx"

    def test_sub_config_override(self):
        cfg = PipelineConfig(
            model=ModelConfig(model_name="custom/model"),
            training=TrainingConfig(iters=500),
        )
        assert cfg.model.model_name == "custom/model"
        assert cfg.training.iters == 500


class TestShortName:
    def test_with_org_prefix(self):
        assert _short_name("google/functiongemma-270m-it") == "functiongemma-270m-it"

    def test_without_org_prefix(self):
        assert _short_name("functiongemma-270m-it") == "functiongemma-270m-it"

    def test_multiple_slashes(self):
        assert _short_name("a/b/c") == "c"


class TestDerivedDefaultPaths:
    """Default paths are derived from the default model_name."""

    def test_model_mlx_path(self):
        cfg = PipelineConfig()
        assert cfg.model.mlx_path == "./functiongemma-270m-it-mlx"

    def test_lora_adapter_path(self):
        cfg = PipelineConfig()
        assert cfg.lora.adapter_path == "adapters-functiongemma-270m-it"

    def test_fuse_path(self):
        cfg = PipelineConfig()
        assert cfg.fuse.fuse_path == "./functiongemma-270m-it-new"

    def test_upload_repo_id(self):
        cfg = PipelineConfig()
        assert cfg.upload.repo_id == "deimagjas/functiongemma-270m-it-mlx"


class TestCustomModelDerivedPaths:
    """Paths derived from a custom model_name."""

    def test_derived_paths(self):
        cfg = PipelineConfig(model=ModelConfig(model_name="org/my-model"))
        assert cfg.model.mlx_path == "./my-model-mlx"
        assert cfg.lora.adapter_path == "adapters-my-model"
        assert cfg.fuse.fuse_path == "./my-model-new"
        assert cfg.upload.repo_id == "deimagjas/my-model-mlx"


class TestExplicitOverridesNotOverwritten:
    """Explicit values must not be overwritten by the validator."""

    def test_explicit_mlx_path(self):
        cfg = PipelineConfig(model=ModelConfig(mlx_path="/custom/mlx"))
        assert cfg.model.mlx_path == "/custom/mlx"

    def test_explicit_adapter_path(self):
        cfg = PipelineConfig(lora=LoraConfig(adapter_path="my-adapters"))
        assert cfg.lora.adapter_path == "my-adapters"

    def test_explicit_fuse_path(self):
        cfg = PipelineConfig(fuse=FuseConfig(fuse_path="/custom/fuse"))
        assert cfg.fuse.fuse_path == "/custom/fuse"

    def test_explicit_repo_id(self):
        cfg = PipelineConfig(upload=UploadConfig(repo_id="user/repo"))
        assert cfg.upload.repo_id == "user/repo"

    def test_env_override_mlx_path(self, monkeypatch):
        monkeypatch.setenv("MODEL_MLX_PATH", "/env/mlx")
        cfg = PipelineConfig(model=ModelConfig())
        assert cfg.model.mlx_path == "/env/mlx"

    def test_env_override_adapter_path(self, monkeypatch):
        monkeypatch.setenv("LORA_ADAPTER_PATH", "env-adapters")
        cfg = PipelineConfig(lora=LoraConfig())
        assert cfg.lora.adapter_path == "env-adapters"
