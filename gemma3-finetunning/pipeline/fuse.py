import subprocess


def fuse_model(model_path: str, adapter_path: str, save_path: str) -> None:
    subprocess.run(
        [
            "python",
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            model_path,
            "--adapter-path",
            adapter_path,
            "--save-path",
            save_path,
        ],
        check=True,
    )
