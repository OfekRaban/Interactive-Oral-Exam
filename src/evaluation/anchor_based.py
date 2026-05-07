"""
Anchor-Based Evaluator — grades answers by comparison to reference anchor examples.

Anchors format::

    {
      "anchors": [
        {"level": "excellent", "answer": "...", "score": 95},
        {"level": "adequate",  "answer": "...", "score": 65},
        {"level": "poor",      "answer": "...", "score": 20}
      ]
    }
"""

import logging
from typing import List

from src.evaluation.llm_engine import LLMEngine
from src.evaluation.prompts import build_anchor_messages
from src.models.evaluation_result import AnchorEvaluationResult
from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)


def _score_to_label(score: float) -> str:
    if score >= 80:
        return "correct"
    elif score >= 40:
        return "partial"
    return "incorrect"


class AnchorBasedEvaluator:
    """
    Evaluates answers by comparing them to a set of calibrated anchor examples.

    Usage::

        anchors_config = {"anchors": [{"level": "excellent", "answer": "...", "score": 95}, ...]}
        ev = AnchorBasedEvaluator(engine, anchors_config)
        results = ev.evaluate_batch(transcript_records)
    """

    def __init__(self, engine: LLMEngine, anchors_config: dict):
        self.engine = engine
        self.anchors = anchors_config["anchors"]
        self.anchor_levels = {a["level"] for a in self.anchors}

    def evaluate_one(self, record: TranscriptRecord) -> AnchorEvaluationResult:
        messages = build_anchor_messages(
            question=record.question,
            answer=record.transcript,
            anchors=self.anchors,
        )
        try:
            data = self.engine.call_json(messages)
            score = max(0.0, min(100.0, float(data.get("score", 0))))
            closest = data.get("closest_anchor", "")
            if closest not in self.anchor_levels:
                closest = self._infer_closest_anchor(score)
            return AnchorEvaluationResult(
                student_id=record.student_id,
                name=record.name,
                score=score,
                closest_anchor=closest,
                feedback=data.get("feedback", ""),
                label=_score_to_label(score),
                anchor_comparisons=data.get("anchor_comparisons", {}),
            )
        except Exception as e:
            logger.error(f"Anchor eval failed for {record.student_id}: {e}", exc_info=True)
            return AnchorEvaluationResult(
                student_id=record.student_id,
                name=record.name,
                score=0.0,
                closest_anchor="",
                feedback="Evaluation failed.",
                label="incorrect",
                error=str(e),
            )

    def evaluate_batch(self, records: List[TranscriptRecord]) -> List[AnchorEvaluationResult]:
        results = []
        for i, record in enumerate(records, 1):
            logger.info(f"Anchor eval [{i}/{len(records)}]: {record.student_id} ({record.name})")
            if record.error:
                results.append(
                    AnchorEvaluationResult(
                        student_id=record.student_id,
                        name=record.name,
                        score=0.0,
                        closest_anchor="",
                        feedback="Transcription failed — cannot evaluate.",
                        label="incorrect",
                        error=record.error,
                    )
                )
            else:
                results.append(self.evaluate_one(record))
        return results

    def _infer_closest_anchor(self, score: float) -> str:
        """Pick the anchor whose score is closest to the given score."""
        best = min(self.anchors, key=lambda a: abs(a["score"] - score))
        return best["level"]
