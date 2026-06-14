import asyncio
import threading
import traceback
from typing import Any

from config import GENERAL_LLM_MODEL


class HuggingFaceChatModel:
    """Lazy local Hugging Face chat model for non-topic reasoning tasks."""

    def __init__(self, model_name: str = GENERAL_LLM_MODEL):
        self.model_name = model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_lock = threading.Lock()
        self.last_error = ""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        try:
            return await asyncio.to_thread(
                self._generate_sync,
                system_prompt,
                user_prompt,
                max_new_tokens,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return ""

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        self.last_error = ""
        self._ensure_model_loaded_sync()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            encoded = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
        except Exception as exc:
            raise RuntimeError(f"chat template failed for {self.model_name}: {exc}") from exc

        import torch

        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        except Exception as exc:
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise RuntimeError(
                f"generation failed for {self.model_name}: {detail}. "
                "This can happen if the model exceeds available memory or the Transformers input format changed."
            ) from exc

        input_length = encoded["input_ids"].shape[-1]
        generated = outputs[0][input_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _ensure_model_loaded_sync(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=False)
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                raise RuntimeError(f"tokenizer load failed for {self.model_name}: {detail}") from exc
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
            if torch.backends.mps.is_available():
                model_kwargs["torch_dtype"] = torch.float16

            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            except Exception as exc:
                detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                raise RuntimeError(f"model load failed for {self.model_name}: {detail}") from exc
            if torch.backends.mps.is_available():
                model = model.to("mps")
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
