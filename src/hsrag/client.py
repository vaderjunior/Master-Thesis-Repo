"""
LLM clients. One interface, two providers.

WHY A BASE CLASS: the retry policy, the fallback discipline, the active_model
stamp and the API log are what every attribution guarantee in this thesis
rests on. They must be byte-identical across providers, or "we switched
provider" silently becomes "we also changed how failures are handled". The
subclasses supply only the wire format: how to build a body, where to send it,
and where the answer sits in the response.

PROVIDERS
  tudagpt - TU Darmstadt HAWKI. Non-standard payload, everything nested under
            "payload", message content as an object. Keep the body minimal:
            adding "stream" or "tools" caused server-side 500s (Phase 0-1).
  peasec  - PEASEC GPU servers behind an Open-WebUI gateway. Standard
            OpenAI /chat/completions. Data stays on PEASEC infrastructure,
            which is what makes it acceptable for DeTox under its Zenodo
            terms (decision Q8).
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

API_LOG = Path("experiments/results/api_log.jsonl")


class ModelUnavailable(Exception):
    """The model is offline or not enabled on this instance.
    Permanent for that model - retrying will not help, another model might."""


class BadRequest(Exception):
    """The server refused this specific request body (4xx that is not 429).

    Deliberately NOT a RequestException, so tenacity does not retry it: the
    same body is refused identically every time, and retrying five times with
    exponential backoff wasted ~30 s per occurrence before this existed.
    """


class TruncatedResponse(Exception):
    """The model hit its token limit mid-answer.

    Its own class because the symptom - unparseable JSON - is identical to a
    model that simply wrote badly, and the two need different fixes. Raising
    here means it appears as a request failure rather than silently inflating
    the reported parse-failure rate.
    """


class BaseClient:
    """Shared retry, fallback, attribution and logging."""

    def __init__(self, models, temperature=1.0, timeout=180,
                 allow_fallback=True, max_tokens=None,
                 url_env=None, token_env=None):
        self.models = models if isinstance(models, list) else [models]
        self.temperature = temperature
        self.timeout = timeout
        self.allow_fallback = allow_fallback
        self.max_tokens = max_tokens
        self.base_url = os.environ[url_env]
        self.token = os.environ[token_env]
        self.active_model = None      # which model actually answered

    # --- provider-specific, overridden ------------------------------------

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _build_body(self, model: str, messages: list[dict]) -> dict:
        raise NotImplementedError

    def _extract_text(self, data: dict) -> str:
        raise NotImplementedError

    # --- shared -----------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        wait=wait_exponential(min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _post(self, body: dict) -> dict:
        """One request. Retries network errors, 429 and 5xx.
        Fails immediately on auth errors, model-unavailable and other 4xx."""
        response = requests.post(
            self._endpoint(),
            headers={"Authorization": f"Bearer {self.token}",
                     "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )

        if response.status_code in (401, 403):
            raise RuntimeError(
                f"Auth error {response.status_code}: check token / API access")

        if (response.status_code >= 400
                and "not available" in response.text.lower()):
            try:
                msg = response.json().get("message", response.text[:200])
            except ValueError:
                msg = response.text[:200]
            raise ModelUnavailable(msg)

        # 429 is genuine rate limiting and SHOULD be retried; other 4xx are
        # about this request and will not fix themselves.
        if 400 <= response.status_code < 500 and response.status_code != 429:
            raise BadRequest(f"{response.status_code}: {response.text[:300]}")

        response.raise_for_status()      # 429/5xx -> retry
        return response.json()

    def _log_call(self, model, prompt_chars, response_chars, latency):
        API_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(API_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": time.time(),
                "provider": self.provider,
                "model": model,
                "prompt_chars": prompt_chars,
                "response_chars": response_chars,
                "latency_s": round(latency, 2),
            }) + "\n")

    def complete(self, messages: list[dict]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "text": "..."}]"""
        errors = []
        for model in self.models:
            body = self._build_body(model, messages)
            try:
                start = time.time()
                data = self._post(body)
                latency = time.time() - start
            except (ModelUnavailable, BadRequest, TruncatedResponse,
                    requests.exceptions.RequestException) as e:
                errors.append(f"{model}: {type(e).__name__}: {e}")
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Model {model} failed and fallback is disabled: {e}"
                    ) from e
                print(f"  [{model}] failed, falling back to next model")
                continue

            self.active_model = model
            text = self._extract_text(data)
            self._log_call(model,
                           sum(len(m["text"]) for m in messages),
                           len(text), latency)
            return text

        raise RuntimeError("All models failed:\n" + "\n".join(errors))


class TUDaGPTClient(BaseClient):
    """TU Darmstadt HAWKI. Non-standard payload; keep it minimal."""

    provider = "tudagpt"

    def __init__(self, models, url_env="TUDAGPT_URL",
                 token_env="TUDAGPT_TOKEN", **kwargs):
        super().__init__(models, url_env=url_env, token_env=token_env, **kwargs)

    def _endpoint(self) -> str:
        return self.base_url          # env var holds the full endpoint

    def _build_body(self, model, messages):
        # Only model, temperature, messages. Adding "stream" or "tools",
        # copied from the web app's own calls, caused server-side 500s.
        return {"payload": {
            "model": model,
            "temperature": self.temperature,
            "messages": [{"role": m["role"], "content": {"text": m["text"]}}
                         for m in messages],
        }}

    def _extract_text(self, data):
        return data["content"]["text"]


class PEASECClient(BaseClient):
    """PEASEC GPU servers via the Open-WebUI gateway. Standard OpenAI shape."""

    provider = "peasec"

    def __init__(self, models, url_env="PEASEC_URL",
                 token_env="PEASEC_TOKEN", **kwargs):
        super().__init__(models, url_env=url_env, token_env=token_env, **kwargs)

    def _endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _build_body(self, model, messages):
        body = {
            "model": model,
            "temperature": self.temperature,
            "messages": [{"role": m["role"], "content": m["text"]}
                         for m in messages],
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        return body

    def _extract_text(self, data):
        choice = data["choices"][0]
        # A truncated answer produces unparseable JSON, which is
        # indistinguishable downstream from a model that wrote badly. Catching
        # it here keeps the reported parse-failure rate honest.
        if choice.get("finish_reason") == "length":
            raise TruncatedResponse(
                f"hit max_tokens={self.max_tokens}; raise it in config")
        return choice["message"]["content"]

    def list_models(self) -> list[str]:
        """GET /models. The TUDaGPT roster drifted silently between phases;
        asking the server beats trusting documentation."""
        r = requests.get(self.base_url.rstrip("/") + "/models",
                         headers={"Authorization": f"Bearer {self.token}"},
                         timeout=30)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))


class MockClient:
    """Canned JSON, no network. Same interface."""

    provider = "mock"

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
        self._calls = 0

    def complete(self, messages: list[dict]) -> str:
        self._calls += 1
        if self.broken and self._calls == 1:
            return "Sure! Here is your answer: {hate: true"
        return json.dumps({
            "reasoning": "The text attacks a protected group.",
            "hate": True,
            "target_group": ["gender"],
            "hate_type": ["explicit"],
            "severity": "medium",
        })


CLIENTS = {"tudagpt": TUDaGPTClient, "peasec": PEASECClient, "mock": MockClient}


def make_client(provider: str, models=None, **kwargs):
    """Provider is selected by config, never by code. A run's provider is
    stamped on every result, so a mid-suite swap is visible rather than a
    silent confound."""
    if provider not in CLIENTS:
        raise ValueError(f"unknown provider '{provider}'; "
                         f"known: {sorted(CLIENTS)}")
    if provider == "mock":
        return MockClient(**kwargs)
    return CLIENTS[provider](models, **kwargs)