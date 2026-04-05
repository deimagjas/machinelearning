import sys
from unittest.mock import MagicMock

# Provide mock modules for external dependencies not installed in the test environment
for mod_name in [
    "mlx_lm",
    "mlx_lm.sample_utils",
    "huggingface_hub",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
