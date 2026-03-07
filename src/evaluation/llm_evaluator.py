import re
import json
import requests

from src.evaluation.base import Evaluator
from src.models.evaluation_result import EvaluationResult


class LLMEvaluator(Evaluator):

    def __init__(self):
        from src.config import settings
        self.api_key = settings.GEMINI_API_KEY
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def evaluate(self, question: str, answer: str) -> EvaluationResult:

        prompt = f"""
        You are an examiner.

        Question:
        {question}

        Student Answer:
        {answer}

        Give:
        - A score from 0 to 100
        - Short feedback

        Return strictly in JSON format:
        {{
            "score": number,
            "feedback": "text"
        }}
        """

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(self.url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Gemini call failed: {response.text}")

        data = response.json()

        content = data["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown code fences if present (e.g. ```json ... ```)
        content = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
        content = re.sub(r"```\s*$", "", content.strip(), flags=re.MULTILINE)

        result_data = json.loads(content)

        return EvaluationResult(
            score=result_data["score"],
            feedback=result_data["feedback"]
        )
