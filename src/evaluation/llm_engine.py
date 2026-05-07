"""
Shared LLM inference engine.

Load once via LLMEngine(...) and pass the instance to all evaluators so the
model weights are held in GPU memory only once across a batch run.
"""

import json
import logging
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
    ):
        logger.info(f"Loading model {model_name} on {device} ...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
        )
        self.gen_config = GenerationConfig(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.05,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        logger.info(f"Model ready: {model_name}")

    def call(self, messages: list) -> str:
        """Run model on a message list, return raw decoded new tokens."""
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                generation_config=self.gen_config,
            )

        input_length = model_inputs["input_ids"].shape[-1]
        new_tokens = output_ids[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def call_json(self, messages: list) -> dict:
        """Call model with a JSON-prefill assistant turn and parse the result."""
        messages_with_prefill = list(messages) + [{"role": "assistant", "content": "{"}]
        content = self.call(messages_with_prefill)
        if not content.lstrip().startswith("{"):
            content = "{" + content
        return self.extract_json(content)

    # ── Static JSON utilities (shared with LocalLLMEvaluator) ────────────────

    @staticmethod
    def extract_json(text: str) -> dict:
        """Three-tier JSON extraction: strict full → strict substring → heuristic."""
        logger.debug(f"Parsing JSON from response ({len(text)} chars)")

        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        pos = 0
        while True:
            start = text.find("{", pos)
            if start == -1:
                break
            end = LLMEngine._find_json_end(text, start)
            if end is not None:
                try:
                    result = json.loads(text[start : end + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    pass
            pos = start + 1

        return LLMEngine._extract_heuristic(text)

    @staticmethod
    def _find_json_end(text: str, start: int) -> Optional[int]:
        depth = 0
        i = start
        while i < len(text):
            ch = text[i]
            if ch == '"':
                i += 1
                while i < len(text):
                    c = text[i]
                    if c == "\\":
                        i += 2
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
        result: dict = {}
        m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if m:
            result["score"] = int(float(m.group(1)))
        m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        if m:
            result["label"] = m.group(1).strip()
        m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if m:
            result["feedback"] = m.group(1).replace('\\"', '"').strip()
        else:
            m = re.search(r'"feedback"\s*:\s*"(.{10,})', text, re.DOTALL)
            if m:
                result["feedback"] = m.group(1).replace('\\"', '"')[:300].strip() + "…"
        return result
