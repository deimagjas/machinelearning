from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings


class ModelConfig(BaseSettings):
    model_config = {"env_prefix": "MODEL_"}

    model_name: str = "google/functiongemma-270m-it"
    quantize: bool = True
    dtype: str = "float16"
    q_bits: PositiveInt = 4
    mlx_path: str = "./gemma3f-mlx"
    cache_dir: str = "./models_cache"


class LoraConfig(BaseSettings):
    model_config = {"env_prefix": "LORA_"}

    num_layers: PositiveInt = 4
    rank: PositiveInt = 4
    scale: float = 20.0
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    adapter_path: str = "adaptersf_gemma3"


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

    fuse_path: str = "./new_functiongemma3"


class UploadConfig(BaseSettings):
    model_config = {"env_prefix": "UPLOAD_"}

    repo_id: str = "deimagjas/functiongemma-3-270m-it-sft"


class PipelineConfig(BaseSettings):
    model_config = {"env_prefix": "PIPELINE_"}

    model: ModelConfig = ModelConfig()
    lora: LoraConfig = LoraConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    fuse: FuseConfig = FuseConfig()
    upload: UploadConfig = UploadConfig()
