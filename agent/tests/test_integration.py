# agent/tests/test_integration.py
import pytest

from agent.graph import run_review
from agent.llm import OllamaClient


@pytest.mark.integration
def test_run_review_against_real_ollama_flags_sql_injection():
    diff = """\
--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 def get_user(user_id):
-    return db.execute("SELECT * FROM users WHERE id = ?", [user_id])
+    return db.execute(f"SELECT * FROM users WHERE id = {user_id}")
"""
    client = OllamaClient(model="qwen2.5:7b")

    verdict, trace = run_review(diff=diff, client=client)

    assert verdict.verdict == "request_changes"
    assert len(verdict.security_issues) >= 1
    assert len(trace.steps) == 4
