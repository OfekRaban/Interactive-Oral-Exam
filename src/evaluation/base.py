from abc import ABC, abstractmethod

from src.models.evaluation_result import EvaluationResult


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, question: str, answer: str) -> EvaluationResult:
        pass
