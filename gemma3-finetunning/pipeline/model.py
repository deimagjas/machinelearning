from pathlib import Path

from mlx_lm import convert, load


def load_model(model_name: str) -> tuple:
    model, tokenizer = load(model_name)
    return model, tokenizer


def convert_model(
    model_name: str,
    mlx_path: str,
    quantize: bool = True,
    dtype: str = "float16",
    q_bits: int = 4,
) -> None:
    if Path(mlx_path).exists():
        print(f"[INFO] Skipping conversion, {mlx_path} already exists.")
        return
    convert(
        model_name,
        quantize=quantize,
        dtype=dtype,
        q_bits=q_bits,
        mlx_path=mlx_path,
    )
