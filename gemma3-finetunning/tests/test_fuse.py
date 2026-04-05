from unittest.mock import patch

from pipeline.fuse import fuse_model


class TestFuseModel:
    @patch("pipeline.fuse.subprocess.run")
    def test_calls_subprocess_with_correct_args(self, mock_run):
        fuse_model(
            model_path="/models/gemma",
            adapter_path="./adapters",
            save_path="./fused_model",
        )

        mock_run.assert_called_once_with(
            [
                "python",
                "-m",
                "mlx_lm",
                "fuse",
                "--model",
                "/models/gemma",
                "--adapter-path",
                "./adapters",
                "--save-path",
                "./fused_model",
            ],
            check=True,
        )
