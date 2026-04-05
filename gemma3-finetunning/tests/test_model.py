from unittest.mock import MagicMock, patch

from pipeline.model import convert_model, load_model


@patch("pipeline.model.load")
def test_load_model(mock_load):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load.return_value = (mock_model, mock_tokenizer)

    model, tokenizer = load_model("google/gemma-3-270m-it")

    mock_load.assert_called_once_with("google/gemma-3-270m-it")
    assert model is mock_model
    assert tokenizer is mock_tokenizer


@patch("pipeline.model.convert")
def test_convert_model(mock_convert, tmp_path):
    mlx_path = str(tmp_path / "gemma3f-mlx")
    convert_model(
        model_name="google/gemma-3-270m-it",
        mlx_path=mlx_path,
        quantize=True,
        dtype="float16",
        q_bits=4,
    )

    mock_convert.assert_called_once_with(
        "google/gemma-3-270m-it",
        quantize=True,
        dtype="float16",
        q_bits=4,
        mlx_path=mlx_path,
    )


@patch("pipeline.model.convert")
def test_convert_model_defaults(mock_convert):
    convert_model(model_name="google/gemma-3-270m-it", mlx_path="./out")

    mock_convert.assert_called_once_with(
        "google/gemma-3-270m-it",
        quantize=True,
        dtype="float16",
        q_bits=4,
        mlx_path="./out",
    )
