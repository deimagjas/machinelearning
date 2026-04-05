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
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")


__all__ = [
    "setup_cache",
    "login_hf",
    "load_model",
    "convert_model",
    "run_inference",
]
