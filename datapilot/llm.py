"""LLM client (Groq). If GROQ_API_KEY isn't set, `available` is False and
callers fall back to their rule-based paths."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

MODEL = os.getenv("DATAPILOT_MODEL", "llama-3.3-70b-versatile")


@dataclass
class LLMResult:
    text: str
    ok: bool
    error: str | None = None


class LLM:
    def __init__(self, model: str = MODEL, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.init_error: str | None = None
        self._client = None
        key = os.getenv("GROQ_API_KEY")
        if not key:
            self.init_error = "GROQ_API_KEY not set"
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=key)
        except ImportError:
            self.init_error = "groq package not installed"

    @property
    def available(self) -> bool:
        return self._client is not None and os.getenv("DATAPILOT_OFFLINE") != "1"

    def complete(self, system: str, prompt: str, max_tokens: int = 2048,
                 json_mode: bool = False) -> LLMResult:
        if not self.available:
            return LLMResult("", False, self.init_error or "LLM disabled")
        import groq

        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
        except groq.AuthenticationError:
            self._client = None
            self.init_error = "Groq rejected the API key"
            return LLMResult("", False, self.init_error)
        except groq.RateLimitError:
            return LLMResult("", False, "Groq rate limit hit, try again in a minute")
        except groq.APIStatusError as e:
            return LLMResult("", False, f"Groq API error {e.status_code}")
        except groq.APIConnectionError:
            return LLMResult("", False, "Couldn't reach Groq")
        return LLMResult((resp.choices[0].message.content or "").strip(), True)

    def complete_json(self, system: str, prompt: str, max_tokens: int = 1024) -> dict | None:
        res = self.complete(system + " Respond with JSON only.", prompt, max_tokens, json_mode=True)
        return extract_json(res.text) if res.ok else None


def extract_json(text: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(candidate)
    except ValueError:
        return None


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
