from src.config import settings
from src.stt.elevenlabs import ElevenLabsSTT
from src.evaluation.local_llm_evaluator import LocalLLMEvaluator
from src.models.evaluation_result import EvaluationResult


class SingleTurnExam:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        self.stt = ElevenLabsSTT(api_key=settings.ELEVENLABS_API_KEY)
        self.evaluator = LocalLLMEvaluator(model_name=model_name)

    def run(self, audio_path: str, question: str) -> EvaluationResult:
        transcription = self.stt.transcribe(audio_path)
        return self.evaluator.evaluate(question=question, answer=transcription.transcript)
