"""
BatchPipeline — top-level orchestrator for the batch evaluation workflow.

Output directory layout
-----------------------
    output/
      raw_files/          downloaded audio/video files (kept persistently)
      transcripts/        {exam_id}_transcripts.json
      results/            {exam_id}_{method}_results.json
                          {exam_id}_tournament_group_rankings.json  (group_ranking mode)
                          {exam_id}_tournament_matchups.json         (legacy modes)

Typical usage
-------------
1. Ingest (download + transcribe):

    pipeline = BatchPipeline(output_dir="output/")
    records = pipeline.ingest("path/to/batch_input.json")
    # Re-run safely: already-downloaded files and existing transcripts are skipped.

2. Evaluate stored transcripts with one or more methods:

    pipeline.evaluate("exam_001", "direct_scoring")
    pipeline.evaluate("exam_001", "rubric",   rubric=rubric_dict)
    pipeline.evaluate("exam_001", "anchor",   anchors=anchors_dict)
    pipeline.evaluate("exam_001", "pairwise")
    pipeline.evaluate("exam_001", "tournament")

3. Full pipeline in one call:

    pipeline.run_pipeline(
        input_path="batch_input.json",
        methods=["direct_scoring", "rubric", "tournament"],
        rubric=rubric_dict,
    )
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from src.batch.processor import BatchProcessor
from src.config import settings
from src.evaluation.anchor_based import AnchorBasedEvaluator
from src.evaluation.direct_scoring import DirectScoringEvaluator
from src.evaluation.llm_engine import LLMEngine
from src.evaluation.pairwise import PairwiseEvaluator
from src.evaluation.rubric_based import RubricBasedEvaluator
from src.evaluation.tournament import TournamentEvaluator
from src.models.student import StudentRecord, TranscriptRecord
from src.storage.result_store import ResultStore
from src.storage.transcript_store import TranscriptStore

logger = logging.getLogger(__name__)

_STT_BACKEND = os.environ.get("STT_BACKEND", "elevenlabs").lower()
ALL_METHODS = ["direct_scoring", "rubric", "anchor", "pairwise", "tournament"]


def _build_stt(backend: str = _STT_BACKEND):
    if backend == "whisper":
        from src.stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT()
    from src.stt.elevenlabs import ElevenLabsSTT
    return ElevenLabsSTT(api_key=settings.ELEVENLABS_API_KEY)


class BatchPipeline:
    """
    Orchestrates the full batch evaluation pipeline.

    Parameters
    ----------
    output_dir:
        Root directory.  Sub-dirs raw_files/, transcripts/, results/ are
        created automatically.
    model_name:
        HuggingFace model ID for the LLM evaluator.
    stt_backend:
        "elevenlabs" (default) or "whisper".
    max_stt_workers:
        Parallel STT workers.  Use 1 for local Whisper, >1 for API.

    Pairwise parameters
    -------------------
    pairwise_batch_size:
        If None, run global all-vs-all.
        If set, shuffle students and run all-vs-all only inside each batch.
    pairwise_shuffle_seed:
        Seed for shuffling before batching (default 42).

    Tournament parameters
    ---------------------
    tournament_mode:
        "group_ranking" (default) — LLM ranks a group of answers at once.
        "round_robin"  — legacy pairwise round-robin.
        "sampled"      — legacy pairwise with random opponent sampling.
    tournament_batch_size:
        For group_ranking: number of students per outer batch (None = one batch).
    tournament_group_size:
        For group_ranking: students per LLM ranking call (default 10).
    tournament_shuffle_seed:
        Shuffle seed before forming batches/groups (default 42).
    tournament_sample_k:
        For legacy "sampled" mode: opponents per student.
    """

    def __init__(
        self,
        output_dir: str = "output",
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        stt_backend: str = _STT_BACKEND,
        max_stt_workers: int = 1,
        # pairwise
        pairwise_batch_size: Optional[int] = None,
        pairwise_shuffle_seed: int = 42,
        # tournament
        tournament_mode: str = "group_ranking",
        tournament_batch_size: Optional[int] = None,
        tournament_group_size: int = 10,
        tournament_shuffle_seed: int = 42,
        tournament_sample_k: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.stt_backend = stt_backend
        self.max_stt_workers = max_stt_workers
        self.pairwise_batch_size = pairwise_batch_size
        self.pairwise_shuffle_seed = pairwise_shuffle_seed
        self.tournament_mode = tournament_mode
        self.tournament_batch_size = tournament_batch_size
        self.tournament_group_size = tournament_group_size
        self.tournament_shuffle_seed = tournament_shuffle_seed
        self.tournament_sample_k = tournament_sample_k

        self.transcript_store = TranscriptStore(
            str(self.output_dir / "transcripts")
        )
        self.result_store = ResultStore(str(self.output_dir / "results"))

        self._engine: Optional[LLMEngine] = None

    @property
    def engine(self) -> LLMEngine:
        """Lazy-load the LLM once and share it across all evaluators."""
        if self._engine is None:
            self._engine = LLMEngine(model_name=self.model_name)
        return self._engine

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, input_path: str) -> List[TranscriptRecord]:
        """
        Load batch_input.json, download files, transcribe, and save transcripts.

        Re-run safe:
        - Files already in raw_files/ are reused without re-downloading.
        - Students whose transcripts are already stored are skipped.

        Returns the complete list of TranscriptRecord objects for the exam
        (both previously stored and newly transcribed).
        """
        batch = self._load_input(input_path)
        exam_id = batch["exam_id"]
        question = batch["question"]
        students = [
            StudentRecord(
                student_id=s["student_id"],
                name=s["name"],
                file_path=s["file"],
                question=s.get("question"),
                metadata=s.get("metadata", {}),
            )
            for s in batch["students"]
        ]

        # Load existing transcripts to find who is already done.
        existing: dict[str, TranscriptRecord] = {}
        try:
            for r in self.transcript_store.load(exam_id):
                existing[r.student_id] = r
            if existing:
                logger.info(
                    f"Skipping {len(existing)} already-transcribed students "
                    f"for exam '{exam_id}'"
                )
        except FileNotFoundError:
            pass

        new_students = [s for s in students if s.student_id not in existing]

        if not new_students:
            logger.info("All students already transcribed — nothing to do.")
            return list(existing.values())

        stt = _build_stt(self.stt_backend)
        processor = BatchProcessor(
            stt,
            raw_files_dir=str(self.output_dir / "raw_files"),
            max_workers=self.max_stt_workers,
        )
        new_records = processor.process_batch(exam_id, question, new_students)

        # Merge in original student order, then save.
        id_to_record = {**existing, **{r.student_id: r for r in new_records}}
        all_records = [
            id_to_record[s.student_id]
            for s in students
            if s.student_id in id_to_record
        ]

        path = self.transcript_store.save(exam_id, all_records)
        ok = sum(1 for r in all_records if not r.error)
        failed = len(all_records) - ok
        logger.info(f"Ingestion complete: {ok} ok, {failed} failed → {path}")
        return all_records

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        exam_id_or_path: str,
        method: str,
        expected_key_points: str = "Not provided",
        rubric: Optional[dict] = None,
        anchors: Optional[dict] = None,
    ) -> Path:
        """
        Run a single evaluation method on stored transcripts.

        exam_id_or_path can be:
        - An exam_id (loads from output_dir/transcripts/).
        - A direct file path to a transcripts JSON.

        Returns the path to the saved results JSON.
        """
        if Path(exam_id_or_path).exists():
            records = self.transcript_store.load_from_file(exam_id_or_path)
            exam_id = records[0].exam_id if records else "unknown"
        else:
            exam_id = exam_id_or_path
            records = self.transcript_store.load(exam_id)

        logger.info(
            f"Evaluating {len(records)} transcripts with method='{method}'"
        )

        if method == "direct_scoring":
            results = DirectScoringEvaluator(
                self.engine, expected_key_points=expected_key_points
            ).evaluate_batch(records)

        elif method == "rubric":
            if rubric is None:
                raise ValueError("rubric dict is required for method='rubric'")
            results = RubricBasedEvaluator(self.engine, rubric).evaluate_batch(records)

        elif method == "anchor":
            if anchors is None:
                raise ValueError("anchors dict is required for method='anchor'")
            results = AnchorBasedEvaluator(self.engine, anchors).evaluate_batch(records)

        elif method == "pairwise":
            ev = PairwiseEvaluator(self.engine)
            results = ev.evaluate_batch(
                records,
                batch_size=self.pairwise_batch_size,
                shuffle_seed=self.pairwise_shuffle_seed,
            )

        elif method == "tournament":
            ev = TournamentEvaluator(
                self.engine,
                mode=self.tournament_mode,
                tournament_batch_size=self.tournament_batch_size,
                tournament_group_size=self.tournament_group_size,
                tournament_shuffle_seed=self.tournament_shuffle_seed,
                sample_k=self.tournament_sample_k,
            )
            rankings, secondary = ev.evaluate_batch(records)
            if self.tournament_mode == "group_ranking":
                self.result_store.save(
                    exam_id, "tournament_group_rankings", secondary
                )
            else:
                self.result_store.save(exam_id, "tournament_matchups", secondary)
            results = rankings

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: {ALL_METHODS}"
            )

        return self.result_store.save(exam_id, method, results)

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        input_path: str,
        methods: Optional[List[str]] = None,
        expected_key_points: str = "Not provided",
        rubric: Optional[dict] = None,
        anchors: Optional[dict] = None,
        skip_ingestion: bool = False,
        transcript_path: Optional[str] = None,
    ) -> dict:
        """
        Run ingest + all requested evaluation methods end-to-end.

        Returns a dict mapping method name → result file path (or error string).
        """
        if methods is None or methods == ["all"]:
            methods = ALL_METHODS

        if "rubric" in methods and rubric is None:
            raise ValueError("rubric dict required for method 'rubric'")
        if "anchor" in methods and anchors is None:
            raise ValueError("anchors dict required for method 'anchor'")

        if skip_ingestion and transcript_path:
            records = self.transcript_store.load_from_file(transcript_path)
            exam_id = records[0].exam_id if records else "unknown"
        else:
            records = self.ingest(input_path)
            exam_id = records[0].exam_id if records else "unknown"

        output_paths: dict = {}
        for method in methods:
            logger.info(f"=== Running evaluation: {method} ===")
            try:
                path = self.evaluate(
                    exam_id,
                    method,
                    expected_key_points=expected_key_points,
                    rubric=rubric,
                    anchors=anchors,
                )
                output_paths[method] = str(path)
                logger.info(f"  → {path}")
            except Exception as e:
                logger.error(f"  ✗ {method} failed: {e}", exc_info=True)
                output_paths[method] = f"ERROR: {e}"

        return output_paths

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_input(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
