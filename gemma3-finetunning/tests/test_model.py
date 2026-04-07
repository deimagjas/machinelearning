from unittest.mock import MagicMock, patch

from pipeline.model import convert_model, load_model


@patch("pipeline.model.load")
def test_load_model(mock_load):
    # Given — a mocked mlx_lm.load that returns a fake model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    # When — load_model is called with a model path
    model, tokenizer = load_model("google/gemma-3-270m-it")

    # Then — mlx_lm.load is invoked once with the exact path and both outputs are forwarded
    mock_load.assert_called_once_with("google/gemma-3-270m-it")
    assert model is mock_model
    assert tokenizer is mock_tokenizer


@patch("pipeline.model.convert")
def test_convert_model(mock_convert, tmp_path):
    # Given — a temporary output path and explicit conversion parameters
    mlx_path = str(tmp_path / "gemma3f-mlx")

    # When — convert_model is called with quantization and dtype options
    convert_model(
        model_name="google/gemma-3-270m-it",
        mlx_path=mlx_path,
        quantize=True,
        dtype="float16",
        q_bits=4,
    )

    # Then — the underlying mlx_lm.convert receives every parameter exactly as given
    mock_convert.assert_called_once_with(
        "google/gemma-3-270m-it",
        quantize=True,
        dtype="float16",
        q_bits=4,
        mlx_path=mlx_path,
    )


@patch("pipeline.model.convert")
def test_convert_model_defaults(mock_convert):
    # Given — only model_name and mlx_path are provided, relying on defaults

    # When — convert_model is called without explicit quantize, dtype, or q_bits
    convert_model(model_name="google/gemma-3-270m-it", mlx_path="./out")

    # Then — the defaults (quantize=True, float16, 4-bit) are forwarded to mlx_lm.convert
    mock_convert.assert_called_once_with(
        "google/gemma-3-270m-it",
        quantize=True,
        dtype="float16",
        q_bits=4,
        mlx_path="./out",
    )
