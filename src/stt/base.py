from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    """
    Represents the result of a speech-to-text transcription.
    """
    transcript: str
    confidence: Optional[float] = None


class SpeechToText(ABC):
    """
    Abstract base class for speech-to-text providers.
    """

    @abstractmethod
    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe an audio file into text.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            TranscriptionResult: Structured transcription result.
        """
        pass
