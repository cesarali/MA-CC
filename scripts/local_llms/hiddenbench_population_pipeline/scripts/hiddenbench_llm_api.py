from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import load_dotenv
from naming_game.potsdam_network import ensure_windows_vpn_bridge

from hiddenbench_common import PipelineError, parse_json_from_text


_FIXED_TEMPERATURE_MODELS = frozenset(
    {"gpt-5", "gpt-5-codex", "gpt-5.5"}
)


def temperature_for_model(model: str, requested: float | None) -> float | None:
    """Return a provider-compatible temperature for a concrete model."""
    model_name = model.rsplit("/", 1)[-1].lower()
    if model_name in _FIXED_TEMPERATURE_MODELS and requested not in {None, 1.0}:
        # These deployments reject any value other than 1. Omitting the field
        # selects that provider default without coupling other models to it.
        return None
    return requested


def load_repository_env() -> Path | None:
    """Load the repository-level .env from any script working directory."""
    starts = (Path.cwd().resolve(), Path(__file__).resolve().parent)
    checked: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in checked:
                continue
            checked.add(directory)
            env_path = directory / ".env"
            if env_path.is_file():
                load_dotenv(env_path, override=False)
                return env_path
    return None


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None
    protocol: str = "responses"
    temperature: float | None = 0.2
    max_output_tokens: int = 7000
    max_retries: int = 4
    timeout_seconds: float = 120.0
    proxy_url: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        model_env: str = "LLM_MODEL",
        default_model: str | None = None,
        temperature_env: str = "LLM_TEMPERATURE",
    ) -> "LLMConfig":
        """Load an OpenAI-compatible configuration without exposing credentials.

        The Potsdam proxy uses the standard Chat Completions shape.  Generic
        ``LLM_*`` variables take precedence so another compatible service can be
        selected without code changes; the university variables are the safe
        project defaults.
        """
        env_path = load_repository_env()
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("POTSDAM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        model = os.getenv(model_env) or os.getenv("LLM_MODEL") or default_model
        if not api_key:
            raise PipelineError(
                "Set LLM_API_KEY (or POTSDAM_API_KEY for the university proxy)."
            )
        if not model:
            raise PipelineError(f"Set {model_env} or LLM_MODEL.")
        temperature_text = os.getenv(temperature_env)
        if temperature_text is None and temperature_env != "LLM_TEMPERATURE":
            temperature_text = os.getenv("LLM_TEMPERATURE")
        temperature_text = (temperature_text or "0.2").strip()
        temperature = (
            None if temperature_text.lower() == "none" else float(temperature_text)
        )
        base_url = (
            os.getenv("LLM_BASE_URL")
            or os.getenv("BASE_POTSDAM_LLM_URL")
            or None
        )
        proxy_url = (
            ensure_windows_vpn_bridge(
                base_url,
                repository_root=env_path.parent if env_path is not None else None,
            )
            if base_url
            else None
        )
        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            protocol=os.getenv("LLM_PROTOCOL", "chat_completions"),
            temperature=temperature,
            max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "7000")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "4")),
            timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            proxy_url=proxy_url,
        )


class LLMClient:
    """
    OpenAI SDK adapter.

    `responses` uses the Responses API. `chat_completions` supports providers that
    expose only an OpenAI-compatible Chat Completions endpoint.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ):
        try:
            from openai import DefaultHttpxClient, OpenAI
        except ImportError as exc:
            raise PipelineError(
                "The MA-CC conda environment needs the `openai` dependency. "
                "Update it with `conda env update -n MA-CC -f environment.yml`."
            ) from exc

        kwargs: dict[str, Any] = {
            "api_key": config.api_key,
            "timeout": config.timeout_seconds,
            # Retry once, in this class, so permanent 4xx errors are not
            # repeated invisibly inside the SDK.
            "max_retries": 0,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.proxy_url:
            kwargs["http_client"] = DefaultHttpxClient(proxy=config.proxy_url)
        self.client = OpenAI(**kwargs)
        self.config = config
        self._progress_callback = progress_callback

    def _report(self, message: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(message)

    @staticmethod
    def _usage_metadata(response: Any) -> dict[str, Any]:
        """Extract provider token accounting when the proxy supplies it."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            value = usage.model_dump(exclude_none=True)
        elif isinstance(usage, Mapping):
            value = dict(usage)
        else:
            value = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
        return {"usage": value}

    def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        last_error: Exception | None = None
        requested_temp = self.config.temperature if temperature is None else temperature
        temp = temperature_for_model(self.config.model, requested_temp)
        max_tokens = max_output_tokens or self.config.max_output_tokens
        attempts_used = 0

        if temp != requested_temp:
            self._report(
                f"Omitting temperature={requested_temp:g} for {self.config.model}; "
                "the deployment requires its fixed default temperature of 1."
            )

        for attempt in range(self.config.max_retries):
            attempts_used = attempt + 1
            attempt_started = time.monotonic()
            try:
                self._report(
                    f"API call started: model={self.config.model}, "
                    f"attempt {attempt + 1}/{self.config.max_retries}, "
                    f"timeout={self.config.timeout_seconds:g}s"
                )
                if self.config.protocol == "responses":
                    kwargs: dict[str, Any] = {
                        "model": self.config.model,
                        "instructions": system,
                        "input": user,
                        "max_output_tokens": max_tokens,
                        "store": False,
                    }
                    if temp is not None:
                        kwargs["temperature"] = temp
                    response = self.client.responses.create(**kwargs)
                    text = response.output_text
                    metadata = {
                        "model": self.config.model,
                        "protocol": "responses",
                        "requested_temperature": requested_temp,
                        "temperature_sent": temp,
                        "response_id": getattr(response, "id", None),
                        "request_id": getattr(response, "_request_id", None),
                    }
                    metadata.update(self._usage_metadata(response))
                elif self.config.protocol == "chat_completions":
                    kwargs = {
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    }
                    if temp is not None:
                        kwargs["temperature"] = temp
                    response = self.client.chat.completions.create(**kwargs)
                    text = response.choices[0].message.content or ""
                    metadata = {
                        "model": self.config.model,
                        "protocol": "chat_completions",
                        "requested_temperature": requested_temp,
                        "temperature_sent": temp,
                        "response_id": getattr(response, "id", None),
                        "request_id": getattr(response, "_request_id", None),
                    }
                    metadata.update(self._usage_metadata(response))
                else:
                    raise PipelineError(
                        "LLM_PROTOCOL must be `responses` or `chat_completions`."
                    )

                if not text.strip():
                    raise PipelineError("The model returned empty text.")
                self._report(
                    "API result received: "
                    f"model={self.config.model}, "
                    f"elapsed={time.monotonic() - attempt_started:.1f}s"
                )
                return text, metadata

            except Exception as exc:
                last_error = exc
                elapsed = time.monotonic() - attempt_started
                status_code = getattr(exc, "status_code", None)
                # Credentials and request-shape errors cannot improve through
                # retries; 429 and transient server/network failures can.
                if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                    break
                if attempt + 1 >= self.config.max_retries:
                    break
                self._report(
                    "API result not received: "
                    f"{type(exc).__name__} after {elapsed:.1f}s; retrying "
                    f"after {min(2**attempt, 8)}s"
                )
                time.sleep(min(2**attempt, 8))

        raise PipelineError(
            "API result not received after "
            f"{attempts_used} attempt(s): {last_error}"
        ) from last_error

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        text, metadata = self.generate_text(
            system=system,
            user=user,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        metadata["raw_text"] = text
        return parse_json_from_text(text), metadata
