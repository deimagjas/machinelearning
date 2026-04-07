from pydantic import BaseModel


class TestPrompt(BaseModel):
    prompt: str