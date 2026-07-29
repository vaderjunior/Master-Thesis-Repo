import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

load_dotenv()


class ModelUnavailable(Exception):
    """The requested model is offline or not enabled on this instance.
    Permanent for that model — retrying won't help, but another model might."""

class BadRequest(Exception):
    """The server refused this specific request body (4xx that is not 429).

    Deliberately NOT a RequestException, so tenacity does not retry it: the
    same body will be refused identically every time, and retrying five times
    with exponential backoff wastes ~30 s per occurrence. Carries the response
    body, because a 422 on specific inputs may be a content filter, and that
    would be systematic missing data rather than a transient error.
    """

class TUDaGPTClient:
    def __init__(
        self,
        models: list[str] | str,
        temperature: float = 1.0,
        timeout: int = 120,
        allow_fallback: bool = True,
    ):
        self.token = os.environ["TUDAGPT_TOKEN"]
        self.url = os.environ["TUDAGPT_URL"]  # full endpoint, incl. /api/ai-req
        self.models = models if isinstance(models, list) else [models]
        self.temperature = temperature
        self.timeout = timeout
        self.allow_fallback = allow_fallback
        self.active_model = None  # which model actually answered the last call

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _post(self, body: dict) -> dict:
        """Send one request. Retries on network errors and 429/500.
        Fails immediately on 401/403 (bad token) and on model-unavailable."""
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Auth error {response.status_code}: check token / API access"
            )

        # model offline / not enabled -> permanent for THIS model, don't retry
        if response.status_code >= 400 and "not available" in response.text.lower():
            try:
                msg = response.json().get("message", response.text[:200])
            except ValueError:
                msg = response.text[:200]
            raise ModelUnavailable(msg)

        # 429 is genuine rate limiting and SHOULD be retried; other 4xx are
        # about this request and will not fix themselves.
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise BadRequest(
                f"{response.status_code}: {response.text[:300]}")

        response.raise_for_status()  # 429/5xx raise -> triggers a retry
        return response.json()

    def _log_call(
        self, model: str, prompt_chars: int, response_chars: int, latency: float
    ) -> None:
        """Append one line per API call, for cost/usage reporting."""
        Path("experiments/results").mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": time.time(),
            "model": model,
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
            "latency_s": round(latency, 2),
        }
        with open("experiments/results/api_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def complete(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"/"user", "text": "..."}]
        Tries each model in order until one succeeds (if allow_fallback).
        Returns the raw model output string."""
        body_messages = [
            {"role": m["role"], "content": {"text": m["text"]}} for m in messages
        ]

        errors = []

        for model in self.models:
            body = {
                "payload": {
                    "model": model,
                    "temperature": self.temperature,
                    "messages": body_messages,
                }
            }

            try:
                start = time.time()
                data = self._post(body)
                latency = time.time() - start

            except (ModelUnavailable, BadRequest,
                    requests.exceptions.RequestException) as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")

                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Model {model} failed and fallback is disabled: {e}"
                    ) from e

                print(f"  [{model}] failed, falling back to next model")
                continue

            self.active_model = model
            text = data["content"]["text"]

            self._log_call(
                model=model,
                prompt_chars=sum(len(m["text"]) for m in messages),
                response_chars=len(text),
                latency=latency,
            )
            return text

        raise RuntimeError("All models failed:\n" + "\n".join(errors))


class MockClient:
    """Drop-in replacement for TUDaGPTClient that returns canned JSON.
    Lets you test the whole pipeline offline and for free."""

    def __init__(self, model="mock", temperature=1.0, broken=False, **kwargs):
        self.model = model
        self.active_model = model
        self.temperature = temperature
        # broken=True fails the FIRST call only, then succeeds. A permanently
        # broken mock can only prove the repair loop gives up; it can never
        # prove the loop recovers, which is the behaviour under test.
        self.broken = broken
        self._calls = 0

    def reset(self) -> None:
        """Re-arm the broken-first-call behaviour between test cases."""
        self._calls = 0

    def complete(self, messages: list[dict]) -> str:
        self._calls += 1

        if self.broken and self._calls == 1:
            return "Sure! Here is your answer: {hate: true"  # invalid JSON

        return json.dumps(
            {
                "reasoning": "The text attacks a protected group.",
                "hate": True,
                "target_group": ["gender"],
                "hate_type": ["explicit"],
                "severity": "medium",
            }
        )