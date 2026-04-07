import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load(
    "mlx-community/functiongemma-270m-it-4bit", # You can replace with any model ID mentioned in the article.
    #tokenizer_config={"trust_remote_code": True}
)

mx.eval(model.parameters())

vocab_size = tokenizer.vocab_size
prompt_length = 1000

mx.random.seed(0)

dummy_tokens = "hello" #mx.random.randint(0, vocab_size, (prompt_length,)).tolist()

tokenizer._eos_token_ids = {}

# warmup
response = generate(
    model, 
    tokenizer, 
    prompt=dummy_tokens, 
    max_tokens=128, 
    verbose=True,
    prefill_step_size=4096 
)

# Actual run
response = generate(
    model, 
    tokenizer, 
    prompt=dummy_tokens, 
    max_tokens=128, 
    verbose=True,
    prefill_step_size=4096  
)