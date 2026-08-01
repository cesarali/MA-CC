import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from hiddenbench_common import (
    allocate_factor_components,
    balanced_type_assignment,
    canonicalize_task,
    source_hidden_texts,
)
from hiddenbench_llm_api import LLMClient, LLMConfig, temperature_for_model


def sample_source_task():
    return {
        "id": 1,
        "name": "sample",
        "description": (
            "You will discuss with three other participants. "
            "You and the other three leaders must decide."
        ),
        "shared_information": ["Shared A"],
        "hidden_information": ["H0", "H1", "H2", "H3"],
        "possible_answers": ["A", "B", "C"],
        "correct_answer": "C",
        "rationale": "test",
    }


def test_canonicalization_preserves_source_and_neutralizes_population():
    source = sample_source_task()
    task = canonicalize_task(source)
    assert task["source_description"] == source["description"]
    assert "three other participants" not in task["scenario_description"].lower()
    assert "other three" not in task["scenario_description"].lower()
    assert source_hidden_texts(task) == ["H0", "H1", "H2", "H3"]


def test_balanced_assignment_covers_all_types():
    labels = balanced_type_assignment(16, 4, seed=12)
    assert set(labels) == {0, 1, 2, 3}
    counts = [labels.count(index) for index in range(4)]
    assert max(counts) - min(counts) <= 1


def test_factor_allocation_covers_every_component():
    components = [
        {
            "evidence_type": evidence_type,
            "component_id": f"{evidence_type}-{component_index}",
            "text": f"fact {evidence_type}-{component_index}",
        }
        for evidence_type in range(4)
        for component_index in range(2)
    ]
    allocation, diagnostics = allocate_factor_components(
        components, 4, seed=3
    )
    assigned = {
        component["component_id"]
        for packet in allocation
        for component in packet
    }
    assert assigned == {
        component["component_id"] for component in components
    }
    assert len(allocation) == 4
    assert all(packet for packet in allocation)
    assert diagnostics["num_components"] == 8


def test_factor_allocation_fills_extra_agents():
    components = [
        {
            "evidence_type": 0,
            "component_id": "0-a",
            "text": "a",
        },
        {
            "evidence_type": 0,
            "component_id": "0-b",
            "text": "b",
        },
    ]
    allocation, _ = allocate_factor_components(components, 6, seed=9)
    assert len(allocation) == 6
    assert all(len(packet) >= 1 for packet in allocation)


def test_university_proxy_configuration_uses_operation_model(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("POTSDAM_API_KEY", "test-key")
    monkeypatch.setenv("BASE_POTSDAM_LLM_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LLM_FACTORIZATION_MODEL", "proxy/gpt-5.5")

    config = LLMConfig.from_env(
        model_env="LLM_FACTORIZATION_MODEL",
        default_model="proxy/default",
    )

    assert config.api_key == "test-key"
    assert config.base_url == "https://proxy.example/v1"
    assert config.protocol == "chat_completions"
    assert config.model == "proxy/gpt-5.5"


def test_operation_temperature_overrides_global_temperature(monkeypatch):
    monkeypatch.setattr("hiddenbench_llm_api.load_repository_env", lambda: None)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("LLM_FACTORIZATION_TEMPERATURE", "0.6")
    monkeypatch.setenv("POTSDAM_API_KEY", "test-key")
    monkeypatch.setenv("BASE_POTSDAM_LLM_URL", "https://proxy.example/v1")

    config = LLMConfig.from_env(
        model_env="LLM_FACTORIZATION_MODEL",
        default_model="proxy/model",
        temperature_env="LLM_FACTORIZATION_TEMPERATURE",
    )

    assert config.temperature == 0.6


def test_operation_temperature_falls_back_to_global_temperature(monkeypatch):
    monkeypatch.setattr("hiddenbench_llm_api.load_repository_env", lambda: None)
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    monkeypatch.delenv("LLM_FACTORIZATION_TEMPERATURE", raising=False)
    monkeypatch.setenv("POTSDAM_API_KEY", "test-key")
    monkeypatch.setenv("BASE_POTSDAM_LLM_URL", "https://proxy.example/v1")

    config = LLMConfig.from_env(
        model_env="LLM_FACTORIZATION_MODEL",
        default_model="proxy/model",
        temperature_env="LLM_FACTORIZATION_TEMPERATURE",
    )

    assert config.temperature == 0.3


def test_gpt5_uses_its_fixed_provider_temperature():
    assert temperature_for_model("microsoft/gpt-5", 0.2) is None
    assert temperature_for_model("microsoft/gpt-5-codex", 0.0) is None
    assert temperature_for_model("microsoft/gpt-5.5", 0.1) is None
    assert temperature_for_model("microsoft/gpt-5", 1.0) == 1.0
    assert temperature_for_model("microsoft/gpt-5.5", 1.0) == 1.0


def test_gpt5_chat_request_omits_incompatible_temperature():
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="response-1",
                _request_id="request-1",
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="API works."))
                ],
                usage=None,
            )

    client = object.__new__(LLMClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    client.config = LLMConfig(
        model="microsoft/gpt-5",
        api_key="test-key",
        protocol="chat_completions",
        temperature=0.2,
        max_retries=1,
    )
    client._progress_callback = None

    text, metadata = client.generate_text(system="system", user="user")

    assert text == "API works."
    assert "temperature" not in captured
    assert metadata["requested_temperature"] == 0.2
    assert metadata["temperature_sent"] is None
