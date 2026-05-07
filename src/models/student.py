from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class StudentRecord:
    student_id: str
    name: str
    file_path: str
    question: Optional[str] = None  # overrides exam-level question when set
    metadata: dict = field(default_factory=dict)


@dataclass
class TranscriptRecord:
    student_id: str
    name: str
    transcript: str
    question: str
    exam_id: str
    confidence: Optional[float] = None
    audio_path: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict = field(default_factory=dict)
