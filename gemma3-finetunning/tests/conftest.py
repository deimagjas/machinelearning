import sys
from unittest.mock import MagicMock

# Mock mlx and mlx_lm modules for Linux CI (mlx is Apple Silicon only)
_mlx_modules = [
    "mlx",
    "mlx.core",
    "mlx.nn",
    "mlx.optimizers",
    "mlx.utils",
    "mlx_lm",
    "mlx_lm.tuner",
    "mlx_lm.tuner.datasets",
]

for mod in _mlx_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
