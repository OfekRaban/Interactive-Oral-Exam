from dataclasses import dataclass


@dataclass
class EvaluationResult:
    score: int
    feedback: str
    label: str
