def __getattr__(name):
    if name == "setup_cache":
        from pipeline.environment import setup_cache
        return setup_cache
    if name == "login_hf":
        from pipeline.environment import login_hf
        return login_hf
    if name == "load_model":
        from pipeline.model import load_model
        return load_model
    if name == "convert_model":
        from pipeline.model import convert_model
        return convert_model
    if name == "run_inference":
        from pipeline.inference import run_inference
        return run_inference
    if name == "setup_adapter":
        from pipeline.lora import setup_adapter
        return setup_adapter
    if name == "apply_lora":
        from pipeline.lora import apply_lora
        return apply_lora
    if name == "load_training_data":
        from pipeline.data import load_training_data
        return load_training_data
    if name == "Metrics":
        from pipeline.train import Metrics
        return Metrics
    if name == "run_training":
        from pipeline.train import run_training
        return run_training
    if name == "fuse_model":
        from pipeline.fuse import fuse_model
        return fuse_model
    if name == "upload_model":
        from pipeline.upload import upload_model
        return upload_model
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")


__all__ = [
    "setup_cache",
    "login_hf",
    "load_model",
    "convert_model",
    "run_inference",
    "setup_adapter",
    "apply_lora",
    "load_training_data",
    "Metrics",
    "run_training",
    "fuse_model",
    "upload_model",
]
