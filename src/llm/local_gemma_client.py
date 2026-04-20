"""Local Gemma inference client using Hugging Face Transformers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LocalGemmaResponse:
    text: str
    model: str
    finish_reason: str = "stop"
    logprobs: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs,
            "usage": self.usage or {},
        }


class LocalGemmaClient:
    """Local Gemma generation client with lazy model download/loading."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_name = cfg.get("model_name") or cfg.get("model") or "google/gemma-4-e4b-it"
        self.device = cfg.get("device", "cuda")
        self.max_new_tokens = int(cfg.get("max_new_tokens", cfg.get("max_tokens", 256)))
        self.max_input_tokens = int(cfg.get("max_input_tokens", 0))
        self.temperature = float(cfg.get("temperature", 0.0))
        self._device_map_cfg: Union[str, Dict[str, Any], None] = cfg.get("device_map", "auto")
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._transformers_ready = False

    def _resolve_device_map(self) -> Union[str, Dict[str, Any], None]:
        """Map model to device(s). ``single`` puts the full model on GPU 0 (no CPU offload)."""
        if not self.device.startswith("cuda"):
            return None
        raw = self._device_map_cfg
        if raw is None or raw == "auto":
            return "auto"
        if raw in ("single", "gpu", "cuda0", "cuda:0"):
            return {"": 0}
        if isinstance(raw, dict):
            return raw
        return "auto"

    def _ensure_loaded(self) -> None:
        if self._transformers_ready:
            return

        hf_home = self.cfg.get("hf_home")
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "Local Gemma dependencies missing. Install: torch transformers accelerate sentencepiece"
            ) from exc

        self._torch = torch
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        device_map = self._resolve_device_map()
        if self.device.startswith("cuda") and device_map is None:
            device_map = "auto"
        load_kw: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            load_kw["device_map"] = device_map
        try:
            logger.info("Loading tokenizer/model locally: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    attn_implementation="sdpa",
                    **load_kw,
                )
            except (TypeError, ValueError, RuntimeError) as sdpa_exc:
                logger.info("Retrying model load without SDPA (%s)", sdpa_exc)
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kw)
        except Exception as exc:
            msg = str(exc).lower()
            if "gated" in msg or "401" in msg or "403" in msg or "access" in msg:
                raise RuntimeError(
                    "Unable to access the requested Gemma model on Hugging Face. "
                    "Run `huggingface-cli login` and make sure your account has access to "
                    f"`{self.model_name}`."
                ) from exc
            raise RuntimeError(
                "Failed to download/load local Gemma model. "
                "Check internet, model id, GPU memory, and Hugging Face auth."
            ) from exc

        self._model.eval()
        self._transformers_ready = True

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        chunks = []
        for msg in messages:
            role = msg.get("role", "user").strip().upper()
            content = msg.get("content", "")
            chunks.append(f"{role}:\n{content}")
        chunks.append("ASSISTANT:\n")
        return "\n\n".join(chunks)

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> LocalGemmaResponse:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        prompt = self._messages_to_prompt(messages)
        tokenize_kwargs = {"return_tensors": "pt"}
        if self.max_input_tokens > 0:
            # Truncate oversized prompts to avoid CUDA OOM on smaller GPUs.
            tokenize_kwargs.update({"truncation": True, "max_length": self.max_input_tokens})
        inputs = self._tokenizer(prompt, **tokenize_kwargs)
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        do_sample = self.temperature > 0
        max_new = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        with self._torch.inference_mode():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[1])
        generated_tokens = gen[0][prompt_len:]
        text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        usage = {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(generated_tokens.shape[0]),
        }
        return LocalGemmaResponse(text=text, model=self.model_name, usage=usage)
