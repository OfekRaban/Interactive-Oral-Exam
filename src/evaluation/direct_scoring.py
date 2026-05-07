"""
Direct Scoring Evaluator — assigns a numeric score 0-100 and a label to each answer.
This is a batch-aware wrapper around the existing LocalLLMEvaluator logic.
"""

import logging
from typing import List, Optional

from src.evaluation.llm_engine import LLMEngine
from src.evaluation.prompts import build_default_messages, VALID_LABELS
from src.models.evaluation_result import DirectScoringResult
from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)


def _score_to_label(score: int) -> str:
    if score >= 80:
        return "correct"
    elif score >= 40:
        return "partial"
    return "incorrect"


class DirectScoringEvaluator:
    """
    Scores each answer independently on a 0-100 scale using an LLM.

    Usage::

        engine = LLMEngine()
        ev = DirectScoringEvaluator(engine, expected_key_points="...")
        results = ev.evaluate_batch(transcript_records)
    """

    def __init__(
        self,
        engine: LLMEngine,
        expected_key_points: str = "Not provided",
    ):
        self.engine = engine
        self.expected_key_points = expected_key_points

    def evaluate_one(self, record: TranscriptRecord) -> DirectScoringResult:
        messages = build_default_messages(
            question=record.question,
            answer=record.transcript,
            expected_key_points=self.expected_key_points,
        )
        try:
            data = self.engine.call_json(messages)
            self._validate(data)
            score = data["score"]
            label = _score_to_label(score)
            llm_label = data.get("label", label)
            if llm_label != label:
                logger.warning(
                    f"[{record.student_id}] LLM label '{llm_label}' overridden to '{label}'"
                )
            return DirectScoringResult(
                student_id=record.student_id,
                name=record.name,
                score=score,
                feedback=data.get("feedback", ""),
                label=label,
            )
        except Exception as e:
            logger.error(f"Failed to score {record.student_id}: {e}", exc_info=True)
            return DirectScoringResult(
                student_id=record.student_id,
                name=record.name,
                score=0,
                feedback="Evaluation failed.",
                label="incorrect",
                error=str(e),
            )

    def evaluate_batch(self, records: List[TranscriptRecord]) -> List[DirectScoringResult]:
        results = []
        for i, record in enumerate(records, 1):
            logger.info(f"Direct scoring [{i}/{len(records)}]: {record.student_id} ({record.name})")
            if record.error:
                results.append(
                    DirectScoringResult(
                        student_id=record.student_id,
                        name=record.name,
                        score=0,
                        feedback="Transcription failed — cannot evaluate.",
                        label="incorrect",
                        error=record.error,
                    )
                )
            else:
                results.append(self.evaluate_one(record))
        return results

    @staticmethod
    def _validate(data: dict) -> None:
        if "score" not in data:
            raise ValueError(f"No score in model output: {data}")
        data["score"] = max(0, min(100, int(data["score"])))
        if not isinstance(data.get("feedback"), str) or not data["feedback"].strip():
            data["feedback"] = "No feedback provided."
        if data.get("label") not in VALID_LABELS:
            data["label"] = _score_to_label(data["score"])
