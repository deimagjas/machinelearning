import os
import pathlib

import huggingface_hub


def setup_cache(cache_dir: str) -> pathlib.Path:
    path = pathlib.Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    return path


def login_hf() -> None:
    huggingface_hub.login()
