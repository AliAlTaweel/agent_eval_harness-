from pydantic import BaseModel

from clinical_review.llm import FakeOllamaClient


class Greeting(BaseModel):
    text: str


def test_fake_client_returns_canned_response_and_usage():
    canned = Greeting(text="hello")
    client = FakeOllamaClient(response=canned, tokens_in=10, tokens_out=3)

    result, usage = client.chat(system="be nice", user="say hi", response_model=Greeting)

    assert result == canned
    assert usage.tokens_in == 10
    assert usage.tokens_out == 3
