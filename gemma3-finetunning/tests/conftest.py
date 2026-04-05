import sys
from unittest.mock import MagicMock

# Mock mlx and mlx_lm modules for Linux CI (mlx is Apple Silicon only)
_mock_modules = [
    "mlx",
    "mlx.core",
    "mlx.nn",
    "mlx.optimizers",
    "mlx.utils",
    "mlx_lm",
    "mlx_lm.sample_utils",
    "mlx_lm.tuner",
    "mlx_lm.tuner.datasets",
    "huggingface_hub",
]

for mod in _mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
