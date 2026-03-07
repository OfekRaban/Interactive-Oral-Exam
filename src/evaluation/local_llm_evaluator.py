import json
from typing import Callable, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.evaluation.base import Evaluator
from src.evaluation.prompts import build_default_messages, VALID_LABELS
from src.models.evaluation_result import EvaluationResult

PromptBuilder = Callable[[str, str], list[dict]]


class LocalLLMEvaluator(Evaluator):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype="auto",
        )
        self.prompt_builder = prompt_builder or build_default_messages

        # Build a fixed GenerationConfig for greedy decoding.
        # Passing this object directly to generate() takes full precedence
        # over the model's bundled config (which ships with do_sample=True).
        self._gen_config = GenerationConfig(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.05,
            max_new_tokens=200,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def evaluate(self, question: str, answer: str) -> EvaluationResult:
        messages = self.prompt_builder(question, answer)

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                generation_config=self._gen_config,
            )

        input_length = model_inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_length:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        result_data = self._extract_json(content)
        self._validate_result(result_data)

        return EvaluationResult(
            score=result_data["score"],
            feedback=result_data["feedback"],
            label=result_data["label"],
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the outermost JSON object from the model's response."""
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in model response:\n{text}")

        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

        raise ValueError(f"Unclosed JSON object in model response:\n{text}")

    @staticmethod
    def _validate_result(result: dict) -> None:
        """Validate that the parsed JSON has the expected fields and types."""
        for field in ("score", "feedback", "label"):
            if field not in result:
                raise ValueError(f"Missing required field '{field}' in model output: {result}")
        if not isinstance(result["score"], int):
            raise ValueError(f"'score' must be an integer, got: {type(result['score'])}")
        if not (0 <= result["score"] <= 100):
            raise ValueError(f"'score' must be between 0 and 100, got: {result['score']}")
        if not isinstance(result["feedback"], str):
            raise ValueError(f"'feedback' must be a string, got: {type(result['feedback'])}")
        if result["label"] not in VALID_LABELS:
            raise ValueError(f"'label' must be one of {VALID_LABELS}, got: {result['label']!r}")
