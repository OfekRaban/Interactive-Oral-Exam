import os

from src.config import settings
from src.evaluation.local_llm_evaluator import LocalLLMEvaluator
from src.models.evaluation_result import EvaluationResult

# Switch backend with:  STT_BACKEND=whisper  python -m src.main ...
# Default is "elevenlabs".
_STT_BACKEND = os.environ.get("STT_BACKEND", "elevenlabs").lower()


def _build_stt():
    if _STT_BACKEND == "whisper":
        from src.stt.whisper_local import WhisperLocalSTT
        return WhisperLocalSTT()
    from src.stt.elevenlabs import ElevenLabsSTT
    return ElevenLabsSTT(api_key=settings.ELEVENLABS_API_KEY)


class SingleTurnExam:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.stt = _build_stt()
        self.evaluator = LocalLLMEvaluator(model_name=model_name)

    def run(self, audio_path: str, question: str) -> EvaluationResult:
        transcription = self.stt.transcribe(audio_path)
        print(f"Transcript: {transcription.transcript}")
        return self.evaluator.evaluate(question=question, answer=transcription.transcript)
