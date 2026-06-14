import asyncio
import re
import threading
from typing import Any

import yaml

from config import TOPIC_PATTERN_BASE_MODEL, TOPIC_PATTERN_MODEL, USE_TOPIC_PATTERN_MODEL


class TopicPatternAgent:
    """Fine-tuned topic/pattern + coaching-step classifier."""

    def __init__(
        self,
        frameworks_path: str = "frameworks.yaml",
        model_name: str = TOPIC_PATTERN_MODEL,
        base_model_name: str = TOPIC_PATTERN_BASE_MODEL,
    ):
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.enabled = USE_TOPIC_PATTERN_MODEL
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str = "cpu"
        self._load_error: str = ""
        self.last_error = ""
        self._load_lock = threading.Lock()
        with open(frameworks_path, "r", encoding="utf-8") as file:
            self.frameworks: dict[str, dict[str, Any]] = yaml.safe_load(file)

    async def analyze(self, question: str) -> dict[str, Any]:
        if not self.enabled or not question.strip():
            self.last_error = "Topic/pattern model is disabled or question is empty."
            return {}

        try:
            output = await asyncio.to_thread(self._generate, question.strip())
        except Exception as exc:
            self._load_error = str(exc)
            self.last_error = str(exc)
            return {}

        parsed = self.parse_output(output)
        framework = self._valid_framework(parsed.get("type", ""))
        steps = parsed.get("steps") or []
        if not framework or not steps:
            self.last_error = f"Could not parse topic model output: {output[:300]}"
            return {}

        self.last_error = ""
        return {
            "type": framework,
            "pattern": parsed.get("type", ""),
            "steps": steps,
            "confidence": 0.85,
            "model": self.model_name,
        }

    def parse_output(self, text: str) -> dict[str, Any]:
        clean = self._extract_assistant_text(text)
        type_match = re.search(r"Type\s*:\s*(.+?)(?:\n|$)", clean, flags=re.IGNORECASE)
        steps_match = re.search(r"Steps\s*:\s*(.+)", clean, flags=re.IGNORECASE | re.DOTALL)

        raw_type = type_match.group(1).strip() if type_match else ""
        raw_steps = steps_match.group(1).strip() if steps_match else ""
        steps = self._split_steps(raw_steps)

        return {"type": raw_type, "steps": steps}

    def _generate(self, question: str) -> str:
        self._ensure_model_loaded_sync()
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=False)

    def _ensure_model_loaded_sync(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer

            adapter_path = snapshot_download(self.model_name, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
            if torch.backends.mps.is_available():
                model_kwargs["torch_dtype"] = torch.float16

            try:
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name, **model_kwargs)
                model = PeftModel.from_pretrained(base_model, adapter_path)
            except Exception as exc:
                self.last_error = f"PEFT load failed: {exc}"
                raise

            if torch.backends.mps.is_available():
                model = model.to("mps")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model

    def _extract_assistant_text(self, text: str) -> str:
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant", 1)[-1]
        if "<|im_end|>" in text:
            text = text.split("<|im_end|>", 1)[0]
        return text.strip()

    def _split_steps(self, text: str) -> list[str]:
        if not text:
            return []

        parts = re.split(r"\s*(?:→|->|,|\n|;)\s*", text)
        steps = []
        for part in parts:
            clean = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", part).strip()
            if clean:
                steps.append(clean)
        return steps[:6]

    def _valid_framework(self, value: str) -> str:
        aliases = {
            "Behavioural": "Behavioral",
            "Product Design": "Product Sense",
            "Data Science": "Technical",
            "AI Engineering": "Technical",
            "Estimation": "Case",
        }
        normalized = aliases.get(value.strip(), value.strip())
        return normalized if normalized in self.frameworks else ""
