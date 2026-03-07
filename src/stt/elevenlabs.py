from src.stt.base import SpeechToText, TranscriptionResult
import requests

class ElevenLabsSTT(SpeechToText):
    """
    Concrete implementation of SpeechToText using ElevenLabs API.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        url = "https://api.elevenlabs.io/v1/speech-to-text"

        headers = {
            "xi-api-key": self.api_key
        }

        with open(audio_path, "rb") as audio_file:
            files = {"file": audio_file}
            data = {"model_id": "scribe_v1"}
            response = requests.post(url, headers=headers, files=files, data=data)

        if response.status_code != 200:
            raise Exception(
                f"STT API call failed with status {response.status_code}: {response.text}"
            )

        data = response.json()

        transcript = data.get("text", "")
        confidence = data.get("language_probability")

        return TranscriptionResult(
            transcript=transcript,
            confidence=confidence
        )