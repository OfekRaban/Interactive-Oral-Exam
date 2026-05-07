import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from src.models.student import TranscriptRecord

logger = logging.getLogger(__name__)


class TranscriptStore:
    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def save(self, exam_id: str, records: List[TranscriptRecord]) -> Path:
        path = self.store_dir / f"{exam_id}_transcripts.json"
        data = {
            "exam_id": exam_id,
            "count": len(records),
            "records": [asdict(r) for r in records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(records)} transcripts to {path}")
        return path

    def load(self, exam_id: str) -> List[TranscriptRecord]:
        path = self.store_dir / f"{exam_id}_transcripts.json"
        return self.load_from_file(str(path))

    def load_from_file(self, path: str) -> List[TranscriptRecord]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_records = data["records"] if isinstance(data, dict) else data
        return [self._dict_to_record(r) for r in raw_records]

    def list_exams(self) -> List[str]:
        return [
            p.stem.replace("_transcripts", "")
            for p in self.store_dir.glob("*_transcripts.json")
        ]

    @staticmethod
    def _dict_to_record(d: dict) -> TranscriptRecord:
        fields = {k: d.get(k) for k in TranscriptRecord.__dataclass_fields__}
        if fields.get("metadata") is None:
            fields["metadata"] = {}
        return TranscriptRecord(**fields)
