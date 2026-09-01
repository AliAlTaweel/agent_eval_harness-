from dataclasses import dataclass

import ollama
from pydantic import BaseModel


@dataclass
class LlmUsage:
    tokens_in: int
    tokens_out: int


class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def chat(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> tuple[BaseModel, LlmUsage]:
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=response_model.model_json_schema(),
        )
        parsed = response_model.model_validate_json(response["message"]["content"])
        usage = LlmUsage(
            tokens_in=response.prompt_eval_count or 0,
            tokens_out=response.eval_count or 0,
        )
        return parsed, usage


class FakeOllamaClient:
    def __init__(self, response: BaseModel, tokens_in: int = 0, tokens_out: int = 0):
        self._response = response
        self._usage = LlmUsage(tokens_in=tokens_in, tokens_out=tokens_out)

    def chat(
        self, system: str, user: str, response_model: type[BaseModel]
    ) -> tuple[BaseModel, LlmUsage]:
        return self._response, self._usage
