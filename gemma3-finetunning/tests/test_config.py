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
        # Given/When — a ModelConfig is created with no arguments
        cfg = ModelConfig()
        # Then — the default model_name points to the functiongemma-270m-it model
        assert cfg.model_name == "google/functiongemma-270m-it"

    def test_quantize(self):
        # Given/When — a ModelConfig is created with no arguments
        cfg = ModelConfig()
        # Then — quantization is enabled by default
        assert cfg.quantize is True

    def test_dtype(self):
        # Given/When — a ModelConfig is created with no arguments
        cfg = ModelConfig()
        # Then — the default dtype is float16
        assert cfg.dtype == "float16"

    def test_q_bits(self):
        # Given/When — a ModelConfig is created with no arguments
        cfg = ModelConfig()
        # Then — the default quantization bit width is 4
        assert cfg.q_bits == 4

    def test_mlx_path(self):
        # Given/When — a ModelConfig is created with no arguments
        cfg = ModelConfig()
        # Then — mlx_path is derived from the model short name with "-mlx" suffix
        assert cfg.mlx_path == "./functiongemma-270m-it-mlx"


class TestLoraConfigDefaults:
    def test_defaults(self):
        # Given/When — a LoraConfig is created with no arguments
        cfg = LoraConfig()
        # Then — all LoRA hyperparameters have sensible defaults (4 layers, rank 4, scale 20, no dropout, empty adapter path)
        assert cfg.num_layers == 4
        assert cfg.rank == 4
        assert cfg.scale == 20.0
        assert cfg.dropout == 0.0
        assert cfg.adapter_path == ""


class TestTrainingConfigDefaults:
    def test_defaults(self):
        # Given/When — a TrainingConfig is created with no arguments
        cfg = TrainingConfig()
        # Then — training defaults are batch_size=1, 200 iterations, eval every 50 steps, gradient checkpointing on, lr=1e-5
        assert cfg.batch_size == 1
        assert cfg.iters == 200
        assert cfg.steps_per_eval == 50
        assert cfg.grad_checkpoint is True
        assert cfg.learning_rate == 1e-5


class TestDataConfigDefaults:
    def test_defaults(self):
        # Given/When — a DataConfig is created with no arguments
        cfg = DataConfig()
        # Then — defaults point to the google/mobile-actions dataset with 70/10 train/val split and prompt/response features
        assert cfg.hf_dataset_path == "google/mobile-actions"
        assert cfg.train_split == "train[:70%]"
        assert cfg.valid_split == "train[-10%:]"
        assert cfg.prompt_feature == "prompt"
        assert cfg.completion_feature == "response"


class TestFuseConfigDefaults:
    def test_defaults(self):
        # Given/When — a FuseConfig is created with no arguments
        cfg = FuseConfig()
        # Then — fuse_path defaults to empty (will be derived by PipelineConfig)
        assert cfg.fuse_path == ""


class TestUploadConfigDefaults:
    def test_defaults(self):
        # Given/When — an UploadConfig is created with no arguments
        cfg = UploadConfig()
        # Then — repo_id defaults to empty (will be derived by PipelineConfig)
        assert cfg.repo_id == ""


class TestEnvVarOverrides:
    def test_model_name_override(self, monkeypatch):
        # Given — the MODEL_MODEL_NAME env var is set to a different model
        monkeypatch.setenv("MODEL_MODEL_NAME", "google/gemma-3-1b-it")
        # When — ModelConfig is created
        cfg = ModelConfig()
        # Then — model_name reflects the environment variable override
        assert cfg.model_name == "google/gemma-3-1b-it"

    def test_q_bits_override(self, monkeypatch):
        # Given — the MODEL_Q_BITS env var is set to 8
        monkeypatch.setenv("MODEL_Q_BITS", "8")
        # When — ModelConfig is created
        cfg = ModelConfig()
        # Then — q_bits is overridden to 8 instead of the default 4
        assert cfg.q_bits == 8

    def test_training_iters_override(self, monkeypatch):
        # Given — the TRAINING_ITERS env var is set to 500
        monkeypatch.setenv("TRAINING_ITERS", "500")
        # When — TrainingConfig is created
        cfg = TrainingConfig()
        # Then — iters is overridden to 500 instead of the default 200
        assert cfg.iters == 500

    def test_learning_rate_override(self, monkeypatch):
        # Given — the TRAINING_LEARNING_RATE env var is set to 0.001
        monkeypatch.setenv("TRAINING_LEARNING_RATE", "0.001")
        # When — TrainingConfig is created
        cfg = TrainingConfig()
        # Then — learning_rate is overridden to 0.001 instead of the default 1e-5
        assert cfg.learning_rate == 0.001

    def test_data_path_override(self, monkeypatch):
        # Given — the DATA_HF_DATASET_PATH env var points to a custom dataset
        monkeypatch.setenv("DATA_HF_DATASET_PATH", "custom/dataset")
        # When — DataConfig is created
        cfg = DataConfig()
        # Then — hf_dataset_path reflects the environment variable
        assert cfg.hf_dataset_path == "custom/dataset"

    def test_lora_rank_override(self, monkeypatch):
        # Given — the LORA_RANK env var is set to 16
        monkeypatch.setenv("LORA_RANK", "16")
        # When — LoraConfig is created
        cfg = LoraConfig()
        # Then — rank is overridden to 16 instead of the default 4
        assert cfg.rank == 16


class TestValidation:
    def test_q_bits_must_be_positive(self):
        # Given/When — ModelConfig is created with q_bits=0
        # Then — validation rejects zero as an invalid quantization bit width
        with pytest.raises(ValidationError):
            ModelConfig(q_bits=0)

    def test_q_bits_rejects_negative(self):
        # Given/When — ModelConfig is created with q_bits=-1
        # Then — validation rejects negative quantization bit widths
        with pytest.raises(ValidationError):
            ModelConfig(q_bits=-1)

    def test_batch_size_must_be_positive(self):
        # Given/When — TrainingConfig is created with batch_size=0
        # Then — validation rejects zero batch size
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_iters_must_be_positive(self):
        # Given/When — TrainingConfig is created with iters=-10
        # Then — validation rejects negative iteration counts
        with pytest.raises(ValidationError):
            TrainingConfig(iters=-10)

    def test_learning_rate_must_be_positive(self):
        # Given/When — TrainingConfig is created with learning_rate=0
        # Then — validation rejects zero learning rate
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)

    def test_dropout_max_bound(self):
        # Given/When — LoraConfig is created with dropout=1.5
        # Then — validation rejects dropout values above the maximum bound
        with pytest.raises(ValidationError):
            LoraConfig(dropout=1.5)

    def test_dropout_min_bound(self):
        # Given/When — LoraConfig is created with dropout=-0.1
        # Then — validation rejects negative dropout values
        with pytest.raises(ValidationError):
            LoraConfig(dropout=-0.1)

    def test_lora_rank_must_be_positive(self):
        # Given/When — LoraConfig is created with rank=0
        # Then — validation rejects zero LoRA rank
        with pytest.raises(ValidationError):
            LoraConfig(rank=0)

    def test_num_layers_must_be_positive(self):
        # Given/When — LoraConfig is created with num_layers=0
        # Then — validation rejects zero LoRA layers
        with pytest.raises(ValidationError):
            LoraConfig(num_layers=0)


class TestPipelineConfig:
    def test_contains_all_sub_configs(self):
        # Given/When — a PipelineConfig is created with defaults
        cfg = PipelineConfig()
        # Then — it contains properly typed instances of all six sub-configs
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.fuse, FuseConfig)
        assert isinstance(cfg.upload, UploadConfig)

    def test_sub_config_defaults_preserved(self):
        # Given/When — a PipelineConfig is created with no overrides
        cfg = PipelineConfig()
        # Then — each sub-config retains its default values including derived paths
        assert cfg.model.model_name == "google/functiongemma-270m-it"
        assert cfg.lora.rank == 4
        assert cfg.training.iters == 200
        assert cfg.data.hf_dataset_path == "google/mobile-actions"
        assert cfg.fuse.fuse_path == "./functiongemma-270m-it-new"
        assert cfg.upload.repo_id == "deimagjas/functiongemma-270m-it-mlx"

    def test_sub_config_override(self):
        # Given — custom model name and training iterations are specified
        # When — PipelineConfig is created with those overrides
        cfg = PipelineConfig(
            model=ModelConfig(model_name="custom/model"),
            training=TrainingConfig(iters=500),
        )
        # Then — only the overridden values change; other defaults remain
        assert cfg.model.model_name == "custom/model"
        assert cfg.training.iters == 500


class TestShortName:
    def test_with_org_prefix(self):
        # Given/When — _short_name receives a model name with an org prefix
        # Then — only the part after the last slash is returned
        assert _short_name("google/functiongemma-270m-it") == "functiongemma-270m-it"

    def test_without_org_prefix(self):
        # Given/When — _short_name receives a model name without any slash
        # Then — the name is returned unchanged
        assert _short_name("functiongemma-270m-it") == "functiongemma-270m-it"

    def test_multiple_slashes(self):
        # Given/When — _short_name receives a path with multiple slashes
        # Then — only the segment after the last slash is returned
        assert _short_name("a/b/c") == "c"


class TestDerivedDefaultPaths:
    """Default paths are derived from the default model_name."""

    def test_model_mlx_path(self):
        # Given/When — PipelineConfig is created with the default model_name
        cfg = PipelineConfig()
        # Then — mlx_path is derived as "./<short-name>-mlx"
        assert cfg.model.mlx_path == "./functiongemma-270m-it-mlx"

    def test_lora_adapter_path(self):
        # Given/When — PipelineConfig is created with the default model_name
        cfg = PipelineConfig()
        # Then — adapter_path is derived as "adapters-<short-name>"
        assert cfg.lora.adapter_path == "adapters-functiongemma-270m-it"

    def test_fuse_path(self):
        # Given/When — PipelineConfig is created with the default model_name
        cfg = PipelineConfig()
        # Then — fuse_path is derived as "./<short-name>-new"
        assert cfg.fuse.fuse_path == "./functiongemma-270m-it-new"

    def test_upload_repo_id(self):
        # Given/When — PipelineConfig is created with the default model_name
        cfg = PipelineConfig()
        # Then — repo_id is derived as "deimagjas/<short-name>-mlx"
        assert cfg.upload.repo_id == "deimagjas/functiongemma-270m-it-mlx"


class TestCustomModelDerivedPaths:
    """Paths derived from a custom model_name."""

    def test_derived_paths(self):
        # Given — a custom model_name "org/my-model" is provided
        # When — PipelineConfig derives all dependent paths from it
        cfg = PipelineConfig(model=ModelConfig(model_name="org/my-model"))
        # Then — all paths use "my-model" as the base name
        assert cfg.model.mlx_path == "./my-model-mlx"
        assert cfg.lora.adapter_path == "adapters-my-model"
        assert cfg.fuse.fuse_path == "./my-model-new"
        assert cfg.upload.repo_id == "deimagjas/my-model-mlx"


class TestExplicitOverridesNotOverwritten:
    """Explicit values must not be overwritten by the validator."""

    def test_explicit_mlx_path(self):
        # Given — an explicit mlx_path is provided to ModelConfig
        # When — PipelineConfig processes the sub-configs
        cfg = PipelineConfig(model=ModelConfig(mlx_path="/custom/mlx"))
        # Then — the explicit mlx_path is preserved, not overwritten by the derived default
        assert cfg.model.mlx_path == "/custom/mlx"

    def test_explicit_adapter_path(self):
        # Given — an explicit adapter_path is provided to LoraConfig
        # When — PipelineConfig processes the sub-configs
        cfg = PipelineConfig(lora=LoraConfig(adapter_path="my-adapters"))
        # Then — the explicit adapter_path is preserved
        assert cfg.lora.adapter_path == "my-adapters"

    def test_explicit_fuse_path(self):
        # Given — an explicit fuse_path is provided to FuseConfig
        # When — PipelineConfig processes the sub-configs
        cfg = PipelineConfig(fuse=FuseConfig(fuse_path="/custom/fuse"))
        # Then — the explicit fuse_path is preserved
        assert cfg.fuse.fuse_path == "/custom/fuse"

    def test_explicit_repo_id(self):
        # Given — an explicit repo_id is provided to UploadConfig
        # When — PipelineConfig processes the sub-configs
        cfg = PipelineConfig(upload=UploadConfig(repo_id="user/repo"))
        # Then — the explicit repo_id is preserved
        assert cfg.upload.repo_id == "user/repo"

    def test_env_override_mlx_path(self, monkeypatch):
        # Given — the MODEL_MLX_PATH env var is set to a custom path
        monkeypatch.setenv("MODEL_MLX_PATH", "/env/mlx")
        # When — PipelineConfig is created with a default ModelConfig
        cfg = PipelineConfig(model=ModelConfig())
        # Then — the env var takes precedence over the derived default
        assert cfg.model.mlx_path == "/env/mlx"

    def test_env_override_adapter_path(self, monkeypatch):
        # Given — the LORA_ADAPTER_PATH env var is set to a custom path
        monkeypatch.setenv("LORA_ADAPTER_PATH", "env-adapters")
        # When — PipelineConfig is created with a default LoraConfig
        cfg = PipelineConfig(lora=LoraConfig())
        # Then — the env var takes precedence over the derived default
        assert cfg.lora.adapter_path == "env-adapters"
