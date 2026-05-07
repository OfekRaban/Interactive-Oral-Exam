"""
BatchProcessor — downloads and transcribes audio/video files for N students.

Downloaded files are saved persistently to `raw_files_dir` so re-runs can
reuse them without hitting Google Drive again.  Students whose transcripts are
already stored are passed in via `skip_student_ids` so ingestion can resume
after interruption.

Uses ThreadPoolExecutor for I/O-bound (API-based) STT and sequential
processing for GPU-local backends.
"""

import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set

from src.audio.loader import download_gdrive_audio, is_gdrive_url
from src.models.student import StudentRecord, TranscriptRecord
from src.stt.base import SpeechToText

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Downloads (if needed) and transcribes audio files for a list of students.

    Parameters
    ----------
    stt:
        An instantiated SpeechToText backend.
    raw_files_dir:
        Directory where downloaded files are saved persistently.
        Subsequent runs reuse existing files instead of re-downloading.
    max_workers:
        Parallel worker count.  Use 1 for GPU-local STT; >1 for API STT.
    """

    def __init__(
        self,
        stt: SpeechToText,
        raw_files_dir: str = "output/raw_files",
        max_workers: int = 1,
    ):
        self.stt = stt
        self.raw_files_dir = Path(raw_files_dir)
        self.raw_files_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

    def process_batch(
        self,
        exam_id: str,
        default_question: str,
        students: List[StudentRecord],
        skip_student_ids: Optional[Set[str]] = None,
    ) -> List[TranscriptRecord]:
        """
        Process all students, returning one TranscriptRecord per student.

        Students in `skip_student_ids` are omitted from the returned list
        (the caller is responsible for merging them back with existing records).
        Failed students are included with `error` set instead of raising.
        """
        to_process = [
            s for s in students
            if not skip_student_ids or s.student_id not in skip_student_ids
        ]
        logger.info(
            f"Processing {len(to_process)}/{len(students)} students "
            f"(workers={self.max_workers})"
        )

        if not to_process:
            return []

        if self.max_workers > 1:
            return self._process_parallel(exam_id, default_question, to_process)
        return self._process_sequential(exam_id, default_question, to_process)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_sequential(
        self,
        exam_id: str,
        default_question: str,
        students: List[StudentRecord],
    ) -> List[TranscriptRecord]:
        results = []
        for i, student in enumerate(students, 1):
            logger.info(f"[{i}/{len(students)}] {student.student_id} ({student.name})")
            results.append(self._process_one(student, default_question, exam_id))
        return results

    def _process_parallel(
        self,
        exam_id: str,
        default_question: str,
        students: List[StudentRecord],
    ) -> List[TranscriptRecord]:
        results_map: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_one, s, default_question, exam_id): s.student_id
                for s in students
            }
            for future in as_completed(futures):
                sid = futures[future]
                results_map[sid] = future.result()
        return [results_map[s.student_id] for s in students]

    def _process_one(
        self,
        student: StudentRecord,
        default_question: str,
        exam_id: str,
    ) -> TranscriptRecord:
        question = student.question or default_question
        local_path: Optional[str] = None
        tmp_path: Optional[str] = None  # only set if we downloaded to a temp file

        try:
            if is_gdrive_url(student.file_path):
                existing = self._find_existing_file(student.student_id)
                if existing:
                    logger.info(
                        f"{student.student_id}: reusing cached file {existing}"
                    )
                    local_path = existing
                else:
                    logger.info(
                        f"{student.student_id}: downloading from Google Drive"
                    )
                    tmp_path = download_gdrive_audio(student.file_path)
                    ext = Path(tmp_path).suffix or ".mp4"
                    persistent = str(
                        self.raw_files_dir / f"{student.student_id}{ext}"
                    )
                    shutil.move(tmp_path, persistent)
                    tmp_path = None  # successfully moved, no cleanup needed
                    local_path = persistent
                    logger.info(
                        f"{student.student_id}: saved to {local_path}"
                    )
            else:
                local_path = student.file_path

            result = self.stt.transcribe(local_path)
            logger.info(
                f"{student.student_id}: {len(result.transcript)} chars transcribed"
            )

            return TranscriptRecord(
                student_id=student.student_id,
                name=student.name,
                transcript=result.transcript,
                question=question,
                exam_id=exam_id,
                confidence=result.confidence,
                audio_path=local_path,
            )

        except Exception as e:
            logger.error(
                f"Failed to process {student.student_id}: {e}", exc_info=True
            )
            return TranscriptRecord(
                student_id=student.student_id,
                name=student.name,
                transcript="",
                question=question,
                exam_id=exam_id,
                error=str(e),
            )

        finally:
            # Only clean up if the download happened but the move did not
            # (e.g., an exception occurred between download and move).
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _find_existing_file(self, student_id: str) -> Optional[str]:
        """Return the path of an already-downloaded file for this student, or None."""
        for p in self.raw_files_dir.iterdir():
            if p.stem == student_id:
                return str(p)
        return None
