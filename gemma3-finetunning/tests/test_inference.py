from unittest.mock import MagicMock, call, patch

from pipeline.inference import run_inference


@patch("pipeline.inference.generate")
@patch("pipeline.inference.make_sampler")
def test_run_inference_applies_chat_template(mock_make_sampler, mock_generate):
    # Given — a mock tokenizer that formats prompts and a mock generator that returns canned text
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<formatted>hello</formatted>"
    mock_model = MagicMock()
    mock_sampler = MagicMock()
    mock_make_sampler.return_value = mock_sampler
    mock_generate.return_value = "response text"

    # When — run_inference is called with a plain text prompt
    result = run_inference(mock_model, mock_tokenizer, "hello")

    # Then — the prompt is wrapped in a chat template, a sampler is created with default temp, and generate receives the formatted prompt
    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    mock_make_sampler.assert_called_once_with(temp=0.2)
    mock_generate.assert_called_once_with(
        mock_model,
        mock_tokenizer,
        prompt="<formatted>hello</formatted>",
        sampler=mock_sampler,
    )
    assert result == "response text"


@patch("pipeline.inference.generate")
@patch("pipeline.inference.make_sampler")
def test_run_inference_custom_temp(mock_make_sampler, mock_generate):
    # Given — mock model/tokenizer/sampler and a custom sampling temperature of 0.8
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_model = MagicMock()
    mock_sampler = MagicMock()
    mock_make_sampler.return_value = mock_sampler
    mock_generate.return_value = "output"

    # When — run_inference is called with sampler_temp=0.8
    result = run_inference(mock_model, mock_tokenizer, "test", sampler_temp=0.8)

    # Then — the sampler is created with the custom temperature and the generated output is returned
    mock_make_sampler.assert_called_once_with(temp=0.8)
    assert result == "output"
