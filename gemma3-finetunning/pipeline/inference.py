from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler


def run_inference(
    model,
    tokenizer,
    prompt: str,
    sampler_temp: float = 0.2,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    sampler = make_sampler(temp=sampler_temp)
    return generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        sampler=sampler,
    )
