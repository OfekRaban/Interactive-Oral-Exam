"""
Tournament / Ranking Evaluator.

Modes
-----
group_ranking (default)
    Split students into outer batches, then into smaller groups.
    For each group, send ALL answers to the LLM in one call and ask it to
    rank them from best to worst.  Rankings are local to each batch.

    Hyperparameters:
        tournament_batch_size  (None = one global batch)
        tournament_group_size  (default 10)
        tournament_shuffle_seed

    Example — 120 students, batch_size=40, group_size=10:
        3 outer batches × 4 groups = 12 LLM calls total.

round_robin (legacy)
    Every student vs every other student via pairwise comparisons.
    N*(N-1)/2 LLM calls.

sampled (legacy)
    Each student plays sample_k random opponents.
    Fewer LLM calls, but ranking is less precise.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from src.evaluation.llm_engine import LLMEngine
from src.evaluation.pairwise import PairwiseEvaluator
from src.evaluation.prompts import build_group_ranking_messages
from src.models.evaluation_result import (
    GroupRankingResult,
    PairwiseResult,
    TournamentRanking,
)
from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)


class TournamentEvaluator:
    """
    Ranks students using one of three tournament modes.

    Usage::

        # group_ranking (default, recommended for large cohorts)
        ev = TournamentEvaluator(engine, tournament_batch_size=40, tournament_group_size=10)
        rankings, group_results = ev.evaluate_batch(records)

        # legacy round-robin
        ev = TournamentEvaluator(engine, mode="round_robin")
        rankings, matchups = ev.evaluate_batch(records)
    """

    def __init__(
        self,
        engine: LLMEngine,
        mode: str = "group_ranking",
        # group_ranking params
        tournament_batch_size: Optional[int] = None,
        tournament_group_size: int = 10,
        tournament_shuffle_seed: int = 42,
        # legacy pairwise params
        sample_k: Optional[int] = None,
        seed: int = 42,
    ):
        self.engine = engine
        self.mode = mode
        self.tournament_batch_size = tournament_batch_size
        self.tournament_group_size = max(2, tournament_group_size)
        self.tournament_shuffle_seed = tournament_shuffle_seed
        # legacy
        self.pairwise = PairwiseEvaluator(engine)
        self.sample_k = sample_k
        self.rng = random.Random(seed)

    def evaluate_batch(
        self, records: List[TranscriptRecord]
    ) -> tuple[List[TournamentRanking], list]:
        """
        Returns
        -------
        rankings : List[TournamentRanking] sorted by rank (best first, rank=1).
        secondary :
            group_ranking mode  → List[GroupRankingResult]
            legacy modes        → List[PairwiseResult]
        """
        valid = [r for r in records if not r.error]
        if len(valid) < 2:
            logger.warning("Need at least 2 valid transcripts for tournament.")
            return self._empty_rankings(records), []

        if self.mode == "group_ranking":
            return self._run_group_ranking(records, valid)
        return self._run_legacy(records, valid)

    # ── group_ranking mode ────────────────────────────────────────────────────

    def _run_group_ranking(
        self,
        all_records: List[TranscriptRecord],
        valid: List[TranscriptRecord],
    ) -> tuple[List[TournamentRanking], List[GroupRankingResult]]:
        rng = random.Random(self.tournament_shuffle_seed)
        shuffled = list(valid)
        rng.shuffle(shuffled)

        batch_size = self.tournament_batch_size or len(shuffled)
        outer_batches = [
            shuffled[i : i + batch_size]
            for i in range(0, len(shuffled), batch_size)
        ]

        all_group_results: List[GroupRankingResult] = []
        # student_id → TournamentRanking (within-batch rank)
        ranking_map: dict[str, TournamentRanking] = {}

        for batch_idx, batch in enumerate(outer_batches, 1):
            batch_id = f"batch_{batch_idx:03d}"
            groups = [
                batch[i : i + self.tournament_group_size]
                for i in range(0, len(batch), self.tournament_group_size)
            ]
            logger.info(
                f"Tournament {batch_id}: {len(batch)} students, "
                f"{len(groups)} groups of ~{self.tournament_group_size}"
            )

            # points[student_id] → float (higher = better ranked in group)
            points: dict[str, float] = {}

            for group_idx, group in enumerate(groups, 1):
                group_id = f"{batch_id}_group_{group_idx:03d}"
                logger.info(
                    f"  Group ranking {group_id}: {len(group)} students"
                )
                gr = self._rank_group(group, batch_id, group_id)
                all_group_results.append(gr)

                if gr.error:
                    # Assign equal points on failure
                    for sid in gr.student_ids:
                        points[sid] = points.get(sid, 0.0) + 0.0
                else:
                    n = len(gr.ranked_student_ids)
                    for pos, sid in enumerate(gr.ranked_student_ids):
                        # Best → n-1 pts, worst → 0 pts
                        points[sid] = points.get(sid, 0.0) + float(n - 1 - pos)

            # Sort batch by points to produce within-batch ranks
            batch_sids = [r.student_id for r in batch]
            sorted_sids = sorted(
                batch_sids,
                key=lambda sid: -points.get(sid, 0.0),
            )
            id_to_name = {r.student_id: r.name for r in batch}
            for rank, sid in enumerate(sorted_sids, 1):
                ranking_map[sid] = TournamentRanking(
                    student_id=sid,
                    name=id_to_name.get(sid, sid),
                    rank=rank,
                    wins=0,
                    losses=0,
                    ties=0,
                    points=points.get(sid, 0.0),
                    total_comparisons=self.tournament_group_size - 1,
                )

        # Build final list preserving all_records order; failed students last
        rankings = []
        for r in all_records:
            if r.student_id in ranking_map:
                rankings.append(ranking_map[r.student_id])
        # Append failed students
        failed_rank = len(rankings) + 1
        for r in all_records:
            if r.error and r.student_id not in ranking_map:
                rankings.append(
                    TournamentRanking(
                        student_id=r.student_id,
                        name=r.name,
                        rank=failed_rank,
                        wins=0,
                        losses=0,
                        ties=0,
                        points=0.0,
                        total_comparisons=0,
                    )
                )
                failed_rank += 1

        return rankings, all_group_results

    def _rank_group(
        self,
        group: List[TranscriptRecord],
        batch_id: str,
        group_id: str,
    ) -> GroupRankingResult:
        student_ids = [r.student_id for r in group]
        student_answers = [
            {"student_id": r.student_id, "name": r.name, "answer": r.transcript}
            for r in group
        ]
        messages = build_group_ranking_messages(group[0].question, student_answers)
        # Add assistant prefill manually so we can capture the raw response.
        messages_with_prefill = list(messages) + [
            {"role": "assistant", "content": "{"}
        ]

        try:
            raw = self.engine.call(messages_with_prefill)
            if not raw.lstrip().startswith("{"):
                raw = "{" + raw

            data = LLMEngine.extract_json(raw)
            ranked_ids: list = data.get("ranking", [])

            # Validate: every ID must be in the group; missing ones appended.
            group_id_set = set(student_ids)
            valid_ranked = [sid for sid in ranked_ids if sid in group_id_set]
            missing = group_id_set - set(valid_ranked)
            if missing:
                logger.warning(
                    f"{group_id}: LLM omitted {len(missing)} student(s) "
                    f"from ranking — appending: {sorted(missing)}"
                )
                valid_ranked.extend(sorted(missing))

            return GroupRankingResult(
                batch_id=batch_id,
                group_id=group_id,
                student_ids=student_ids,
                ranked_student_ids=valid_ranked,
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )

        except Exception as e:
            logger.error(
                f"Group ranking failed for {group_id}: {e}", exc_info=True
            )
            return GroupRankingResult(
                batch_id=batch_id,
                group_id=group_id,
                student_ids=student_ids,
                ranked_student_ids=student_ids,  # fallback: original order
                reasoning="",
                raw_response="",
                error=str(e),
            )

    # ── legacy pairwise modes ─────────────────────────────────────────────────

    def _run_legacy(
        self,
        all_records: List[TranscriptRecord],
        valid: List[TranscriptRecord],
    ) -> tuple[List[TournamentRanking], List[PairwiseResult]]:
        matchups = self._run_matchups(valid)
        rankings = self._compute_rankings(all_records, matchups)
        return rankings, matchups

    def _run_matchups(self, records: List[TranscriptRecord]) -> List[PairwiseResult]:
        pairs = self._select_pairs(records)
        total = len(pairs)
        results = []
        for i, (a, b) in enumerate(pairs, 1):
            logger.info(
                f"Tournament [{i}/{total}]: {a.student_id} vs {b.student_id}"
            )
            results.append(self.pairwise.compare(a, b))
        return results

    def _select_pairs(
        self, records: List[TranscriptRecord]
    ) -> List[tuple[TranscriptRecord, TranscriptRecord]]:
        if self.mode == "round_robin":
            return [
                (records[i], records[j])
                for i in range(len(records))
                for j in range(i + 1, len(records))
            ]
        # sampled mode
        k = self.sample_k or max(1, len(records) // 2)
        pairs: set = set()
        for record in records:
            opponents = [r for r in records if r.student_id != record.student_id]
            chosen = self.rng.sample(opponents, min(k, len(opponents)))
            for opp in chosen:
                key = tuple(sorted([record.student_id, opp.student_id]))
                if key not in pairs:
                    pairs.add(key)
        id_to_record = {r.student_id: r for r in records}
        return [(id_to_record[a], id_to_record[b]) for a, b in pairs]

    def _compute_rankings(
        self,
        all_records: List[TranscriptRecord],
        matchups: List[PairwiseResult],
    ) -> List[TournamentRanking]:
        wins: dict = defaultdict(int)
        losses: dict = defaultdict(int)
        ties: dict = defaultdict(int)

        for m in matchups:
            if m.winner == "A":
                wins[m.student_a_id] += 1
                losses[m.student_b_id] += 1
            elif m.winner == "B":
                wins[m.student_b_id] += 1
                losses[m.student_a_id] += 1
            else:
                ties[m.student_a_id] += 1
                ties[m.student_b_id] += 1

        id_to_name = {r.student_id: r.name for r in all_records}
        rankings = []
        for record in all_records:
            sid = record.student_id
            w, l, t = wins[sid], losses[sid], ties[sid]
            rankings.append(
                TournamentRanking(
                    student_id=sid,
                    name=id_to_name.get(sid, sid),
                    rank=0,
                    wins=w,
                    losses=l,
                    ties=t,
                    points=w + 0.5 * t,
                    total_comparisons=w + l + t,
                )
            )

        rankings.sort(key=lambda r: (-r.points, -r.wins, r.losses))
        for i, r in enumerate(rankings, 1):
            r.rank = i
        return rankings

    @staticmethod
    def _empty_rankings(records: List[TranscriptRecord]) -> List[TournamentRanking]:
        return [
            TournamentRanking(
                student_id=r.student_id,
                name=r.name,
                rank=i + 1,
                wins=0,
                losses=0,
                ties=0,
                points=0.0,
                total_comparisons=0,
            )
            for i, r in enumerate(records)
        ]
