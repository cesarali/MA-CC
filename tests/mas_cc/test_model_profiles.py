import asyncio
import shutil
import subprocess
import warnings
import zipfile
from pathlib import Path

import pytest

from mas_cc.llm_runtime.config import LLMProviderConfig
from mas_cc.llm_runtime.messages import Message
from mas_cc.llm_runtime.providers import (
    CompletionRequest,
    ModelProfile,
    ModelProfileOverrideWarning,
    ModelProfileRegistry,
    ProviderError,
    TemperatureRule,
    apply_model_profile,
    create_default_provider_registry,
    create_llm_provider,
    default_model_profile_registry,
)
from scripts.exploratory.llm_providers.probe_model_parameters import (
    ProbeResult,
    _scrub,
    derive_temperature_rule,
    main as probe_main,
)
from scripts.exploratory.llm_providers import probe_model_parameters


class _Response:
    status_code = 200
    headers = {}

    def json(self):
        return {
            "id": "profile-test",
            "model": "test-model",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def raise_for_status(self):
        return None


class _Session:
    def __init__(self):
        self.payloads = []

    def post(self, _url, **kwargs):
        self.payloads.append(kwargs["json"])
        return _Response()

    def close(self):
        return None


def _request(*, temperature=0.0, seed=7):
    return CompletionRequest(
        (Message("system", "Be brief."), Message("user", "hi")),
        temperature=temperature,
        max_output_tokens=16,
        seed=seed,
    )


def _profile(mode, *, value=None, model="test-model", **kwargs):
    return ModelProfile(
        provider_type="openai",
        model=model,
        family="test",
        temperature=TemperatureRule(mode, value),
        probe_source="probe",
        **kwargs,
    )


def test_exact_lookup_returns_checked_in_profile():
    profile = default_model_profile_registry().get("university", "microsoft/gpt-4o")
    assert profile.model == "microsoft/gpt-4o"
    assert profile.probe_source == "probe"
    assert profile.temperature == TemperatureRule("any")


def test_unknown_lookup_is_visibly_inferred():
    profile = default_model_profile_registry().get("university", "vendor/new-chat-model")
    assert profile.probe_source == "inferred"
    assert profile.family == "unknown"


def test_fixed_temperature_rewrites_and_records_the_override():
    profile = _profile("fixed", value=1.0, model="fixed-test-model")
    with pytest.warns(ModelProfileOverrideWarning, match="temperature=0.0 -> 1.0"):
        adjusted = apply_model_profile(_request(), profile)
    assert adjusted.temperature == 1.0
    details = adjusted.metadata["model_profile_adjustments"]
    assert details["changes"]["temperature"] == {"requested": 0.0, "sent": 1.0}


def test_override_warning_is_emitted_once_per_model_rule():
    profile = _profile("fixed", value=1.0, model="warning-once-test-model")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        apply_model_profile(_request(temperature=0.0), profile)
        apply_model_profile(_request(temperature=0.5), profile)
    matching = [item for item in captured if item.category is ModelProfileOverrideWarning]
    assert len(matching) == 1


def test_any_temperature_leaves_the_request_object_untouched():
    request = _request(temperature=0.3)
    assert apply_model_profile(request, _profile("any")) is request


def test_seed_is_dropped_and_recorded_when_unsupported():
    with pytest.warns(ModelProfileOverrideWarning, match="seed=7 -> None"):
        adjusted = apply_model_profile(
            _request(), _profile("any", model="seedless-test-model", supports_seed=False)
        )
    assert adjusted.seed is None
    assert adjusted.metadata["model_profile_adjustments"]["changes"]["seed"]["sent"] is None


def test_omit_profile_reaches_adapter_without_a_temperature_field():
    session = _Session()
    registry = ModelProfileRegistry([_profile("omit", model="omit-test-model")])
    provider = create_llm_provider(
        LLMProviderConfig(
            type="openai", model="omit-test-model", credentials_env="TEST_API_KEY"
        ),
        environment={"TEST_API_KEY": "not-a-secret"},
        session=session,
        profile_registry=registry,
    )
    with pytest.warns(ModelProfileOverrideWarning, match=r"temperature=0\.0 -> None"):
        asyncio.run(provider.complete(_request()))
    assert "temperature" not in session.payloads[0]


def test_an_unrepresentable_output_field_fails_before_http_dispatch():
    session = _Session()
    registry = ModelProfileRegistry(
        [_profile("any", max_output_tokens_field="max_completion_tokens")]
    )
    provider = create_llm_provider(
        LLMProviderConfig(type="openai", model="test-model", credentials_env="TEST_API_KEY"),
        environment={"TEST_API_KEY": "not-a-secret"},
        session=session,
        profile_registry=registry,
    )
    with pytest.raises(ProviderError) as captured:
        asyncio.run(provider.complete(_request()))
    assert captured.value.code == "unsupported_model_parameter"
    assert session.payloads == []


def test_profile_json_round_trip():
    original = _profile(
        "fixed",
        value=1.0,
        supports_seed=False,
        supports_system_messages=True,
        notes=("observed",),
    )
    assert ModelProfile.from_mapping(original.to_dict()) == original


def test_all_shipped_entries_parse_and_name_known_provider_types():
    profiles = default_model_profile_registry().known()
    known_provider_types = set(create_default_provider_registry().names())
    assert profiles
    assert {profile.provider_type for profile in profiles} <= known_provider_types


LITELLM_FIXED_TEMPERATURE_ERROR = (
    "litellm.UnsupportedParamsError: gpt-5 models (including gpt-5-codex) don't support "
    "temperature=0.0. Only temperature=1 is supported. For gpt-5.1, temperature is "
    "supported when reasoning_effort='none'. To drop unsupported params set "
    "litellm.drop_params = True. Received Model Group=microsoft/gpt-5.4-nano"
)


def test_probe_derives_fixed_rule_from_real_litellm_failure_fixture():
    results = {
        "temperature_0": ProbeResult("temperature_0", 400, False, LITELLM_FIXED_TEMPERATURE_ERROR),
        "temperature_1": ProbeResult("temperature_1", 200, True),
        "temperature_omitted": ProbeResult("temperature_omitted", 200, True),
    }
    assert derive_temperature_rule(results) == TemperatureRule("fixed", 1.0)


def test_probe_derives_any_when_all_temperature_variants_succeed():
    results = {
        name: ProbeResult(name, 200, True)
        for name in ("temperature_0", "temperature_1", "temperature_omitted")
    }
    assert derive_temperature_rule(results) == TemperatureRule("any")


def test_probe_dry_run_lists_only_selected_models_without_network(tmp_path, capsys):
    models = tmp_path / "models.txt"
    models.write_text("# selected\nmicrosoft/gpt-5-mini\n\nmicrosoft/gpt-4o\n", encoding="utf-8")
    output = tmp_path / "profiles.json"
    assert probe_main(["--models-file", str(models), "--output", str(output)]) == 0
    captured = capsys.readouterr().out
    assert "No network calls made" in captured
    assert "microsoft/gpt-4o" in captured
    assert "microsoft/gpt-5-mini" in captured
    assert not output.exists()


def test_probe_list_phase_writes_advertised_models_sorted_offline(
    tmp_path, monkeypatch
):
    class FakeProvider:
        async def discover_models(self):
            return ("z/model", "a/model")

        def close(self):
            return None

    monkeypatch.setattr(probe_model_parameters, "_provider", lambda *_: FakeProvider())
    output = tmp_path / "models.txt"
    assert probe_main(["--list-models", str(output), "--confirm-live-calls"]) == 0
    assert output.read_text(encoding="utf-8") == "a/model\nz/model\n"


def test_probe_scrubs_api_keys_and_bearer_tokens():
    secret = "super-secret-api-key"
    text = _scrub(
        {"message": f"request failed for {secret}", "authorization": "Bearer abc.def"},
        (secret,),
    )
    assert secret not in text
    assert "abc.def" not in text
    assert text.count("<redacted>") == 2


def test_built_wheel_contains_model_profile_catalogue(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(root / "pyproject.toml", source / "pyproject.toml")
    shutil.copy2(root / "README.md", source / "README.md")
    shutil.copytree(
        root / "src",
        source / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the isolated wheel-content check")
    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--offline",
            "--no-config",
            "--out-dir",
            str(wheel_dir),
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert "mas_cc/llm_runtime/providers/model_profiles.json" in archive.namelist()
