"""
Pairwise Evaluator — given two transcripts, asks the LLM which answer is better.

Used directly or as the building block for TournamentEvaluator.
"""

import logging
import random
from typing import List, Optional

from src.evaluation.llm_engine import LLMEngine
from src.evaluation.prompts import build_pairwise_messages
from src.models.evaluation_result import PairwiseResult
from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)

VALID_WINNERS = {"A", "B", "tie"}


class PairwiseEvaluator:
    """
    Compares two answers and determines which demonstrates better understanding.

    Usage::

        ev = PairwiseEvaluator(engine)

        # Single comparison
        result = ev.compare(record_a, record_b)

        # Batch — global all-vs-all
        results = ev.evaluate_batch(records)

        # Batch — batched all-vs-all (only compares within each batch)
        results = ev.evaluate_batch(records, batch_size=30, shuffle_seed=42)
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine

    def compare(
        self,
        record_a: TranscriptRecord,
        record_b: TranscriptRecord,
        batch_id: Optional[str] = None,
    ) -> PairwiseResult:
        messages = build_pairwise_messages(
            question=record_a.question,
            answer_a=record_a.transcript,
            answer_b=record_b.transcript,
        )
        try:
            data = self.engine.call_json(messages)
            winner = str(data.get("winner", "tie")).strip().upper()
            if winner not in VALID_WINNERS:
                winner = "tie"
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.5, min(1.0, confidence))
            return PairwiseResult(
                student_a_id=record_a.student_id,
                student_b_id=record_b.student_id,
                winner=winner,
                confidence=confidence,
                reasoning=data.get("reasoning", ""),
                batch_id=batch_id,
            )
        except Exception as e:
            logger.error(
                f"Pairwise compare failed ({record_a.student_id} vs "
                f"{record_b.student_id}): {e}",
                exc_info=True,
            )
            return PairwiseResult(
                student_a_id=record_a.student_id,
                student_b_id=record_b.student_id,
                winner="tie",
                confidence=0.5,
                reasoning=f"Comparison failed: {e}",
                batch_id=batch_id,
            )

    def evaluate_batch(
        self,
        records: List[TranscriptRecord],
        batch_size: Optional[int] = None,
        shuffle_seed: int = 42,
    ) -> List[PairwiseResult]:
        """
        Run pairwise comparisons across a list of student transcripts.

        Parameters
        ----------
        records:
            All student transcripts (students with errors are skipped).
        batch_size:
            None  → global all-vs-all (every student compared to every other).
            int   → shuffle students, split into batches of this size, then run
                    all-vs-all only *inside* each batch.  No cross-batch
                    comparisons are made.
        shuffle_seed:
            RNG seed used when batch_size is set.

        Returns a flat list of PairwiseResult objects.  Each result's
        ``batch_id`` is set when batching is active (e.g. "batch_001").
        """
        valid = [r for r in records if not r.error]

        if batch_size is None:
            return self._run_all_pairs(valid, batch_id=None)

        rng = random.Random(shuffle_seed)
        shuffled = list(valid)
        rng.shuffle(shuffled)

        batches = [
            shuffled[i : i + batch_size]
            for i in range(0, len(shuffled), batch_size)
        ]
        total_pairs = sum(len(b) * (len(b) - 1) // 2 for b in batches)
        logger.info(
            f"Pairwise batched: {len(batches)} batches, "
            f"{total_pairs} total comparisons"
        )

        results = []
        done = 0
        for idx, batch in enumerate(batches, 1):
            bid = f"batch_{idx:03d}"
            pairs = self._run_all_pairs(batch, batch_id=bid, start_count=done + 1, total=total_pairs)
            done += len(pairs)
            results.extend(pairs)
        return results

    def _run_all_pairs(
        self,
        records: List[TranscriptRecord],
        batch_id: Optional[str],
        start_count: int = 1,
        total: Optional[int] = None,
    ) -> List[PairwiseResult]:
        pairs = [
            (records[i], records[j])
            for i in range(len(records))
            for j in range(i + 1, len(records))
        ]
        if total is None:
            total = len(pairs)
        results = []
        for k, (a, b) in enumerate(pairs, start_count):
            bid_label = f" batch={batch_id}" if batch_id else ""
            logger.info(
                f"Pairwise [{k}/{total}]{bid_label}: "
                f"{a.student_id} vs {b.student_id}"
            )
            results.append(self.compare(a, b, batch_id=batch_id))
        return results
