from harness.judge import FakeHallucinationJudge, JudgeVerdict


def test_fake_judge_returns_canned_verdict():
    canned = JudgeVerdict(hallucinated=True, reasoning="claim not present in diff")
    judge = FakeHallucinationJudge(response=canned)

    result = judge.judge(source_material="diff text", claim="uses eval() unsafely")

    assert result == canned
