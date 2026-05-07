import json
import re
from typing import Callable, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.evaluation.base import Evaluator
from src.evaluation.prompts import build_default_messages, VALID_LABELS
from src.models.evaluation_result import EvaluationResult

PromptBuilder = Callable[[str, str], list[dict]]


def _score_to_label(score: int) -> str:
    """Deterministically derive label from numeric score, matching prompts.py rubric."""
    if score >= 80:
        return "correct"
    elif score >= 40:
        return "partial"
    return "incorrect"


class LocalLLMEvaluator(Evaluator):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        prompt_builder: Optional[PromptBuilder] = None,
        engine=None,  # Optional[LLMEngine] — share a pre-loaded engine across evaluators
    ):
        if engine is not None:
            # Reuse a shared LLMEngine (avoids loading the model a second time).
            self.tokenizer = engine.tokenizer
            self.model = engine.model
            self._gen_config = engine.gen_config
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            # Build a fixed GenerationConfig for greedy decoding.
            # Passing this object directly to generate() takes full precedence
            # over the model's bundled config (which ships with do_sample=True).
            self._gen_config = GenerationConfig(
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.05,
                max_new_tokens=1024,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.prompt_builder = prompt_builder or build_default_messages

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

        # The assistant prefill "{" is part of the prompt (consumed tokens),
        # so the decoded output starts after it — prepend it back.
        if not content.lstrip().startswith("{"):
            content = "{" + content

        result_data = self._extract_json(content)
        self._validate_result(result_data)

        # Override label deterministically from score to prevent LLM inconsistency.
        llm_label = result_data["label"]
        correct_label = _score_to_label(result_data["score"])
        if llm_label != correct_label:
            print(
                f"[WARN] LLM label '{llm_label}' inconsistent with "
                f"score={result_data['score']} — overriding to '{correct_label}'"
            )

        return EvaluationResult(
            score=result_data["score"],
            feedback=result_data["feedback"],
            label=correct_label,
        )

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Extract the first valid JSON object from model response text.
        Handles extra prose before/after the JSON, and braces inside strings.
        Falls back to regex-based field extraction if no valid JSON is found.
        """
        print(f"[DIAG] Evaluator response  : {len(text)} chars")

        # Pass 1: try strict parse of the whole response.
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                print("[DIAG] JSON parse          : strict (full text)")
                return result
        except json.JSONDecodeError:
            pass

        # Pass 2: find every '{' and scan for the matching '}',
        # properly skipping over string contents so braces inside
        # quoted strings don't confuse the depth counter.
        pos = 0
        while True:
            start = text.find("{", pos)
            if start == -1:
                break
            end = LocalLLMEvaluator._find_json_end(text, start)
            if end is not None:
                candidate = text[start : end + 1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        print("[DIAG] JSON parse          : strict (substring)")
                        return result
                except json.JSONDecodeError:
                    pass
            pos = start + 1

        # Pass 3: heuristic field extraction.
        print("[DIAG] JSON parse          : fallback heuristic")
        return LocalLLMEvaluator._extract_heuristic(text)

    @staticmethod
    def _find_json_end(text: str, start: int):
        """
        Return the index of the closing '}' that matches the '{' at `start`,
        correctly skipping over string literals (including escape sequences).
        Returns None if no matching brace is found.
        """
        depth = 0
        i = start
        while i < len(text):
            ch = text[i]
            if ch == '"':
                # Skip the entire string literal.
                i += 1
                while i < len(text):
                    c = text[i]
                    if c == "\\":
                        i += 2  # escaped character — skip both
                        continue
                    if c == '"':
                        break
                    i += 1
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        return None

    @staticmethod
    def _extract_heuristic(text: str) -> dict:
        """
        Last-resort extraction: pull score, label, and feedback via regex.
        Never raises — fills in safe defaults for any field that cannot be found.
        """
        result = {}
        recovered = []

        # score
        m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            result["score"] = int(float(m.group(1)))
            recovered.append("score")

        # label — optional; derived from score in _validate_result if absent
        m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        if m:
            result["label"] = m.group(1).strip()
            recovered.append("label")

        # feedback — try closed string first, then recover partial unclosed string
        m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if m:
            result["feedback"] = m.group(1).replace('\\"', '"').strip()
            recovered.append("feedback")
        else:
            # Model opened the feedback string but never closed it — grab what we can.
            m = re.search(r'"feedback"\s*:\s*"(.{10,})', text, re.DOTALL)
            if m:
                raw = m.group(1).replace('\\"', '"')
                result["feedback"] = raw[:300].strip() + "…"
                recovered.append("feedback(partial)")

        print(f"[DIAG] Heuristic recovered : {recovered}")
        return result

    @staticmethod
    def _validate_result(result: dict) -> None:
        """
        Normalise and validate the parsed result in-place.
        Fills in safe defaults for missing fields rather than raising,
        as long as score is present or can be defaulted.
        """
        # score — must exist; only hard failure
        if "score" not in result:
            raise ValueError(f"Cannot recover score from model output: {result}")
        try:
            result["score"] = int(result["score"])
        except (TypeError, ValueError):
            raise ValueError(f"'score' must be an integer, got: {result['score']!r}")
        result["score"] = max(0, min(100, result["score"]))

        # feedback — use default if missing or not a string
        if not isinstance(result.get("feedback"), str) or not result["feedback"].strip():
            result["feedback"] = "Evaluation parsed from partial model output."
            print("[DIAG] feedback            : using default (missing/malformed)")

        # label — derive deterministically from score if missing or invalid
        if result.get("label") not in VALID_LABELS:
            derived = _score_to_label(result["score"])
            print(f"[DIAG] label               : derived from score ({result['score']} → {derived})")
            result["label"] = derived
