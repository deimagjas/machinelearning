import os
import pathlib
from inference import model, tokenizer
from mlx_lm import generate

project_root = pathlib.Path.cwd()

cache_dir = project_root / "models_cache"
cache_dir.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(cache_dir)
print(f"La variable de entorno HF_HOME se ha establecido en: {os.environ['HF_HOME']}")

def main():
    prompt = "create a list of steps in order to help someone that has an anxiety attack"
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=prompt, verbose=True)
 

if __name__ == "__main__":
    main()
