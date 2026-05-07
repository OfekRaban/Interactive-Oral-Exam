"""
Rubric-Based Evaluator — scores each answer per criterion defined in a rubric dict.

Rubric format::

    {
      "criteria": [
        {"name": "Accuracy", "description": "...", "max_score": 40},
        {"name": "Completeness", "description": "...", "max_score": 30},
        ...
      ]
    }
"""

import logging
from typing import List

from src.evaluation.llm_engine import LLMEngine
from src.evaluation.prompts import build_rubric_messages
from src.models.evaluation_result import (
    RubricCriterionResult,
    RubricEvaluationResult,
)
from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)


def _normalized_to_label(normalized: float) -> str:
    if normalized >= 80:
        return "correct"
    elif normalized >= 40:
        return "partial"
    return "incorrect"


class RubricBasedEvaluator:
    """
    Evaluates answers against a structured rubric with multiple weighted criteria.
    Returns per-criterion scores plus an overall normalized score (0-100).

    Usage::

        rubric = {"criteria": [{"name": "X", "description": "...", "max_score": 50}, ...]}
        ev = RubricBasedEvaluator(engine, rubric)
        results = ev.evaluate_batch(transcript_records)
    """

    def __init__(self, engine: LLMEngine, rubric: dict):
        self.engine = engine
        self.rubric = rubric
        self.criteria = rubric["criteria"]
        self.max_total = sum(c["max_score"] for c in self.criteria)

    def evaluate_one(self, record: TranscriptRecord) -> RubricEvaluationResult:
        messages = build_rubric_messages(
            question=record.question,
            answer=record.transcript,
            rubric_criteria=self.criteria,
        )
        try:
            data = self.engine.call_json(messages)
            criteria_results = self._parse_criteria(data)
            total = sum(c.score for c in criteria_results)
            normalized = round((total / self.max_total) * 100, 1) if self.max_total else 0.0
            return RubricEvaluationResult(
                student_id=record.student_id,
                name=record.name,
                criteria=criteria_results,
                total_score=total,
                max_score=self.max_total,
                normalized_score=normalized,
                feedback=data.get("total_feedback", ""),
                label=_normalized_to_label(normalized),
            )
        except Exception as e:
            logger.error(f"Rubric eval failed for {record.student_id}: {e}", exc_info=True)
            return RubricEvaluationResult(
                student_id=record.student_id,
                name=record.name,
                criteria=[],
                total_score=0,
                max_score=self.max_total,
                normalized_score=0.0,
                feedback="Evaluation failed.",
                label="incorrect",
                error=str(e),
            )

    def evaluate_batch(self, records: List[TranscriptRecord]) -> List[RubricEvaluationResult]:
        results = []
        for i, record in enumerate(records, 1):
            logger.info(f"Rubric eval [{i}/{len(records)}]: {record.student_id} ({record.name})")
            if record.error:
                results.append(
                    RubricEvaluationResult(
                        student_id=record.student_id,
                        name=record.name,
                        criteria=[],
                        total_score=0,
                        max_score=self.max_total,
                        normalized_score=0.0,
                        feedback="Transcription failed — cannot evaluate.",
                        label="incorrect",
                        error=record.error,
                    )
                )
            else:
                results.append(self.evaluate_one(record))
        return results

    def _parse_criteria(self, data: dict) -> List[RubricCriterionResult]:
        raw = data.get("criteria", [])
        out = []
        name_to_max = {c["name"]: c["max_score"] for c in self.criteria}
        for item in raw:
            name = item.get("name", "Unknown")
            max_s = item.get("max_score") or name_to_max.get(name, 10)
            score = max(0.0, min(float(max_s), float(item.get("score", 0))))
            out.append(
                RubricCriterionResult(
                    name=name,
                    score=score,
                    max_score=float(max_s),
                    feedback=item.get("feedback", ""),
                )
            )
        # Fill in any criteria the model missed
        returned_names = {c.name for c in out}
        for c in self.criteria:
            if c["name"] not in returned_names:
                out.append(
                    RubricCriterionResult(
                        name=c["name"],
                        score=0.0,
                        max_score=float(c["max_score"]),
                        feedback="Not evaluated.",
                    )
                )
        return out
