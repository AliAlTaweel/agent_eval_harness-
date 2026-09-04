from typing import Literal

from pydantic import BaseModel

EncounterType = Literal["new_patient", "follow_up"]


class Finding(BaseModel):
    element: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class CompletenessFindings(BaseModel):
    issues: list[Finding] = []


class ComplianceFindings(BaseModel):
    issues: list[Finding] = []


class CodingClarityFindings(BaseModel):
    issues: list[Finding] = []


class Verdict(BaseModel):
    completeness_issues: list[Finding] = []
    compliance_issues: list[Finding] = []
    coding_clarity_issues: list[Finding] = []
    verdict: Literal["approve", "needs_revision"]
