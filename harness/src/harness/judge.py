import ollama
from pydantic import BaseModel

JUDGE_SYSTEM_PROMPT = (
    "You are an evaluator checking whether a claim made by an AI code "
    "reviewer is actually supported by the provided source material (a "
    "diff or tool output). Respond with hallucinated=true if the claim "
    "describes something not present in or not supported by the source "
    "material, otherwise hallucinated=false. Give brief reasoning."
)


class JudgeVerdict(BaseModel):
    hallucinated: bool
    reasoning: str


class HallucinationJudge:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def judge(self, source_material: str, claim: str) -> JudgeVerdict:
        user_content = f"SOURCE MATERIAL:\n{source_material}\n\nCLAIM:\n{claim}"
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            format=JudgeVerdict.model_json_schema(),
        )
        return JudgeVerdict.model_validate_json(response["message"]["content"])


class FakeHallucinationJudge:
    def __init__(self, response: JudgeVerdict):
        self._response = response

    def judge(self, source_material: str, claim: str) -> JudgeVerdict:
        return self._response
