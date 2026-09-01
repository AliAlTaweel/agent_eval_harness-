from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    file: str
    line: int | None = None
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class SecurityFindings(BaseModel):
    issues: list[Issue] = []


class StyleFindings(BaseModel):
    issues: list[Issue] = []


class TestCoverageFindings(BaseModel):
    has_gap: bool
    explanation: str = ""


class Verdict(BaseModel):
    security_issues: list[Issue] = []
    style_issues: list[Issue] = []
    test_coverage_gap: bool
    verdict: Literal["approve", "request_changes"]
