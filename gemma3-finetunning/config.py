from pydantic import Field, PositiveInt, model_validator
from pydantic_settings import BaseSettings


def _short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


class ModelConfig(BaseSettings):
    model_config = {"env_prefix": "MODEL_"}

    model_name: str = "google/functiongemma-270m-it"
    quantize: bool = True
    dtype: str = "float16"
    q_bits: PositiveInt = 4
    mlx_path: str = ""
    cache_dir: str = "./models_cache"

    @model_validator(mode="after")
    def set_derived_defaults(self):
        short = _short_name(self.model_name)
        if not self.mlx_path:
            self.mlx_path = f"./{short}-mlx"
        return self


class LoraConfig(BaseSettings):
    model_config = {"env_prefix": "LORA_"}

    num_layers: PositiveInt = 4
    rank: PositiveInt = 4
    scale: float = 20.0
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    adapter_path: str = ""


class TrainingConfig(BaseSettings):
    model_config = {"env_prefix": "TRAINING_"}

    batch_size: PositiveInt = 1
    iters: PositiveInt = 200
    steps_per_eval: PositiveInt = 50
    grad_checkpoint: bool = True
    learning_rate: float = Field(default=1e-5, gt=0)


class DataConfig(BaseSettings):
    model_config = {"env_prefix": "DATA_"}

    hf_dataset_path: str = "google/mobile-actions"
    train_split: str = "train[:70%]"
    valid_split: str = "train[-10%:]"
    prompt_feature: str = "prompt"
    completion_feature: str = "response"


class FuseConfig(BaseSettings):
    model_config = {"env_prefix": "FUSE_"}

    fuse_path: str = ""


class UploadConfig(BaseSettings):
    model_config = {"env_prefix": "UPLOAD_"}

    repo_id: str = ""


class PipelineConfig(BaseSettings):
    model_config = {"env_prefix": "PIPELINE_"}

    model: ModelConfig = ModelConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    fuse: FuseConfig = FuseConfig()
    upload: UploadConfig = UploadConfig()

    @model_validator(mode="after")
    def set_derived_paths(self):
        short = _short_name(self.model.model_name)
        if not self.lora.adapter_path:
            self.lora.adapter_path = f"adapters-{short}"
        if not self.fuse.fuse_path:
            self.fuse.fuse_path = f"./{short}-new"
        if not self.upload.repo_id:
            self.upload.repo_id = f"deimagjas/{short}-mlx"
        return self
