#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm.auto import tqdm

from hiddenbench_common import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_CANONICAL_TASKS,
    PipelineError,
    ValidationError,
    append_jsonl,
    deduplicate_texts,
    extract_tasks_payload,
    normalized_text,
    select_tasks,
    task_id,
    write_json,
)
from hiddenbench_llm_api import LLMClient, LLMConfig


GENERATOR_SYSTEM = """
You create controlled scientific benchmark annotations. Return only valid JSON.
Do not solve the task for the participant. Do not add facts. The supplied correct
answer is audit information and must never be leaked into a generated observation.
""".strip()

VERIFIER_SYSTEM = """
You are a strict benchmark auditor. Return only valid JSON. Reject any candidate
when semantic preservation, answer non-leakage, or separation from other evidence
types is uncertain. False rejection is preferable to benchmark contamination.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an LLM generator and an independent verification pass to create "
            "reusable HiddenBench paraphrase pools and factorization annotations."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CANONICAL_TASKS,
        help="canonical task file (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
    )
    parser.add_argument(
        "--mode",
        choices=["paraphrases", "factorizations", "both"],
        default="both",
    )
    parser.add_argument(
        "--paraphrases-per-type",
        type=int,
        default=10,
    )
    parser.add_argument("--paraphrase-batch-size", type=int, default=12)
    parser.add_argument("--max-paraphrase-rounds", type=int, default=10)
    parser.add_argument("--factorization-alternatives", type=int, default=4)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--task-ids", type=int, nargs="*")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue filling existing output files.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if a paraphrase pool does not reach the requested size.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hidden_entries(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for index, value in enumerate(task["hidden_information"]):
        if isinstance(value, Mapping):
            result.append(
                {
                    "evidence_type": int(value.get("evidence_type", index)),
                    "source_text": str(value["source_text"]),
                }
            )
        else:
            result.append(
                {"evidence_type": index, "source_text": str(value)}
            )
    return result


def task_context(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id(task),
        "name": task["name"],
        "description": task.get(
            "scenario_description", task.get("description", "")
        ),
        "shared_information": task["shared_information"],
        "all_hidden_information": [
            entry["source_text"] for entry in hidden_entries(task)
        ],
        "possible_answers": task["possible_answers"],
        "correct_answer_for_audit_only": task["correct_answer"],
    }


def paraphrase_generation_prompt(
    task: Mapping[str, Any],
    evidence_type: int,
    source_text: str,
    count: int,
    avoid_texts: Sequence[str],
) -> str:
    context = task_context(task)
    context.update(
        {
            "target_evidence_type": evidence_type,
            "target_source_text": source_text,
            "existing_variants_to_avoid": list(avoid_texts[-30:]),
        }
    )
    return f"""
Generate {count} distinct paraphrases of the target private fact.

Requirements:
1. Preserve every answer-relevant proposition in the source.
2. Add no new entity, relation, time, quantity, certainty, cause, or consequence.
3. Do not state or hint at the correct answer.
4. Do not import information from any other hidden fact.
5. Change wording and syntax enough to create linguistic diversity.
6. Each variant must stand alone.

Return exactly:
{{
  "variants": [
    {{
      "text": "...",
      "variation_strategy": "lexical|syntactic|referential|discourse"
    }}
  ]
}}

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def paraphrase_verification_prompt(
    task: Mapping[str, Any],
    evidence_type: int,
    source_text: str,
    candidates: Sequence[str],
) -> str:
    context = task_context(task)
    context.update(
        {
            "target_evidence_type": evidence_type,
            "target_source_text": source_text,
            "candidates": [
                {"candidate_index": index, "text": text}
                for index, text in enumerate(candidates)
            ],
        }
    )
    return f"""
Audit each candidate paraphrase.

For every candidate return:
- entailed_by_source
- source_entailed_by_candidate
- adds_answer_relevant_information
- leaks_correct_answer
- overlaps_other_hidden_information
- acceptable

`acceptable` is true only when both entailment checks are true and all three
contamination checks are false.

Return exactly:
{{
  "verdicts": [
    {{
      "candidate_index": 0,
      "entailed_by_source": true,
      "source_entailed_by_candidate": true,
      "adds_answer_relevant_information": false,
      "leaks_correct_answer": false,
      "overlaps_other_hidden_information": false,
      "acceptable": true,
      "notes": "..."
    }}
  ]
}}

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def factorization_generation_prompt(
    task: Mapping[str, Any],
    evidence_type: int,
    source_text: str,
    alternatives: int,
    max_components: int,
) -> str:
    context = task_context(task)
    context.update(
        {
            "target_evidence_type": evidence_type,
            "target_source_text": source_text,
        }
    )
    return f"""
Propose up to {alternatives} meaningful factorizations of the target private fact.

A factorization divides one source fact into 2 to {max_components} informational
components that may be distributed across agents.

A valid factorization must satisfy:
1. Components jointly reconstruct the complete source fact.
2. No single component is equivalent to the complete source fact.
3. Components add no fact not entailed by the source or scenario description.
4. Components do not reveal the correct answer.
5. The split is semantic/inferential, not arbitrary sentence chopping.
6. Mark `factorizable` false when no defensible decomposition exists.

Return exactly:
{{
  "factorizable": true,
  "alternatives": [
    {{
      "components": [
        {{
          "text": "...",
          "role": "entity|relation|constraint|observation|bridge"
        }}
      ],
      "reconstruction_rule": "...",
      "why_meaningful": "..."
    }}
  ],
  "non_factorizable_reason": null
}}

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def factorization_verification_prompt(
    task: Mapping[str, Any],
    evidence_type: int,
    source_text: str,
    alternatives: Sequence[Mapping[str, Any]],
) -> str:
    context = task_context(task)
    context.update(
        {
            "target_evidence_type": evidence_type,
            "target_source_text": source_text,
            "candidate_factorizations": list(alternatives),
        }
    )
    return f"""
Audit every candidate factorization.

Return exactly:
{{
  "verdicts": [
    {{
      "candidate_index": 0,
      "jointly_reconstructs_source": true,
      "every_component_supported": true,
      "each_component_individually_insufficient": true,
      "no_answer_leakage": true,
      "no_other_hidden_information_imported": true,
      "meaningful_not_arbitrary": true,
      "acceptable": true,
      "quality_score": 85,
      "notes": "..."
    }}
  ]
}}

`acceptable` must be false if any required property is false.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
""".strip()


def load_or_initialize(
    path: Path,
    *,
    kind: str,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and resume:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("kind") != kind:
            raise ValidationError(
                f"{path} is not a {kind} annotation file."
            )
        return value
    if path.exists() and not overwrite:
        raise PipelineError(
            f"{path} exists. Use --resume or --overwrite."
        )
    return {
        "schema_version": "1.0",
        "kind": kind,
        "tasks": {},
    }


def accepted_variant_texts(record: Mapping[str, Any]) -> list[str]:
    return [
        str(item["text"])
        for item in record.get("variants", [])
        if isinstance(item, Mapping)
        and item.get("accepted", True)
        and isinstance(item.get("text"), str)
    ]


def generate_paraphrases(
    tasks: Sequence[Mapping[str, Any]],
    generator_client: LLMClient,
    verifier_client: LLMClient,
    output_path: Path,
    audit_path: Path,
    *,
    target: int,
    batch_size: int,
    max_rounds: int,
    resume: bool,
    overwrite: bool,
    require_complete: bool,
) -> None:
    payload = load_or_initialize(
        output_path,
        kind="paraphrase_pool",
        resume=resume,
        overwrite=overwrite,
    )
    payload["target_per_evidence_type"] = target
    payload["generator_model"] = generator_client.config.model
    payload["verifier_model"] = verifier_client.config.model

    for task in tasks:
        tid = str(task_id(task))
        task_record = payload["tasks"].setdefault(
            tid,
            {"name": task["name"], "evidence_types": {}},
        )

        evidence_progress = tqdm(
            hidden_entries(task),
            desc=f"task {tid} paraphrases",
            unit="evidence",
            leave=False,
        )
        for entry in evidence_progress:
            evidence_type = int(entry["evidence_type"])
            source_text = entry["source_text"]
            record = task_record["evidence_types"].setdefault(
                str(evidence_type),
                {
                    "source_text": source_text,
                    "variants": [],
                },
            )
            accepted = accepted_variant_texts(record)
            seen = {normalized_text(source_text)}
            seen.update(normalized_text(text) for text in accepted)
            evidence_progress.set_postfix(
                evidence=evidence_type,
                accepted=f"{len(accepted)}/{target}",
                stage="generation",
            )
            evidence_progress.refresh()

            tqdm.write(
                f"task {tid} evidence {evidence_type}: "
                f"{len(accepted)}/{target} accepted paraphrases"
            )

            for generation_round in range(max_rounds):
                if len(accepted) >= target:
                    break
                requested = min(
                    batch_size,
                    max(batch_size, 2 * (target - len(accepted))),
                )
                evidence_progress.set_postfix(
                    evidence=evidence_type,
                    accepted=f"{len(accepted)}/{target}",
                    stage=f"generator round {generation_round + 1}",
                )
                evidence_progress.refresh()
                generated, generation_meta = generator_client.generate_json(
                    system=GENERATOR_SYSTEM,
                    user=paraphrase_generation_prompt(
                        task,
                        evidence_type,
                        source_text,
                        requested,
                        accepted,
                    ),
                )
                variants = generated.get("variants", [])
                candidates = deduplicate_texts(
                    str(item.get("text", ""))
                    for item in variants
                    if isinstance(item, Mapping)
                )
                candidates = [
                    text
                    for text in candidates
                    if normalized_text(text) not in seen
                ]
                if not candidates:
                    append_jsonl(
                        audit_path,
                        {
                            "kind": "paraphrase_generation",
                            "task_id": task_id(task),
                            "evidence_type": evidence_type,
                            "round": generation_round,
                            "accepted": 0,
                            "metadata": generation_meta,
                            "warning": "No new unique candidates.",
                        },
                    )
                    continue

                evidence_progress.set_postfix(
                    evidence=evidence_type,
                    accepted=f"{len(accepted)}/{target}",
                    stage="verifier",
                )
                evidence_progress.refresh()
                verified, verification_meta = verifier_client.generate_json(
                    system=VERIFIER_SYSTEM,
                    user=paraphrase_verification_prompt(
                        task,
                        evidence_type,
                        source_text,
                        candidates,
                    ),
                )
                verdicts = {
                    int(item["candidate_index"]): item
                    for item in verified.get("verdicts", [])
                    if isinstance(item, Mapping)
                    and "candidate_index" in item
                }

                newly_accepted = 0
                for index, text in enumerate(candidates):
                    verdict = verdicts.get(index, {})
                    accepted_flag = bool(verdict.get("acceptable", False))
                    if accepted_flag and normalized_text(text) not in seen:
                        variant_id = f"{evidence_type}-{len(record['variants']):03d}"
                        record["variants"].append(
                            {
                                "variant_id": variant_id,
                                "text": text,
                                "accepted": True,
                                "generation_metadata": generation_meta,
                                "verification": verdict,
                                "verification_metadata": verification_meta,
                            }
                        )
                        accepted.append(text)
                        seen.add(normalized_text(text))
                        newly_accepted += 1
                        if len(accepted) >= target:
                            break

                append_jsonl(
                    audit_path,
                    {
                        "kind": "paraphrase_round",
                        "task_id": task_id(task),
                        "evidence_type": evidence_type,
                        "round": generation_round,
                        "candidate_count": len(candidates),
                        "accepted_count": newly_accepted,
                        "generation_metadata": generation_meta,
                        "verification_metadata": verification_meta,
                    },
                )
                write_json(output_path, payload, overwrite=True)
                tqdm.write(
                    f"task {tid} evidence {evidence_type}: round "
                    f"{generation_round + 1} verification complete; "
                    f"accepted {len(accepted)}/{target}"
                )

            record["complete"] = len(accepted) >= target
            record["accepted_count"] = len(accepted)
            write_json(output_path, payload, overwrite=True)
            evidence_progress.set_postfix(
                evidence=evidence_type,
                accepted=f"{len(accepted)}/{target}",
                stage="checkpoint",
            )
            tqdm.write(
                f"checkpoint: task {tid}, evidence {evidence_type}, "
                f"paraphrases {len(accepted)}/{target}"
            )
            if require_complete and not record["complete"]:
                raise ValidationError(
                    f"Task {tid}, evidence type {evidence_type}: "
                    f"only {len(accepted)}/{target} paraphrases passed."
                )
        evidence_progress.close()


def normalize_factorization(
    alternative: Mapping[str, Any],
    evidence_type: int,
    alternative_index: int,
) -> dict[str, Any]:
    components = []
    for component_index, component in enumerate(
        alternative.get("components", [])
    ):
        if not isinstance(component, Mapping):
            continue
        text = str(component.get("text", "")).strip()
        if not text:
            continue
        components.append(
            {
                "component_id": (
                    f"{evidence_type}-{alternative_index}-{component_index}"
                ),
                "text": text,
                "role": component.get("role"),
            }
        )
    return {
        "factorization_id": f"{evidence_type}-{alternative_index}",
        "components": components,
        "reconstruction_rule": alternative.get("reconstruction_rule"),
        "why_meaningful": alternative.get("why_meaningful"),
    }


def generate_factorizations(
    tasks: Sequence[Mapping[str, Any]],
    generator_client: LLMClient,
    verifier_client: LLMClient,
    output_path: Path,
    audit_path: Path,
    *,
    alternatives: int,
    max_components: int,
    resume: bool,
    overwrite: bool,
) -> None:
    payload = load_or_initialize(
        output_path,
        kind="factorization_pool",
        resume=resume,
        overwrite=overwrite,
    )
    payload["generator_model"] = generator_client.config.model
    payload["verifier_model"] = verifier_client.config.model

    for task in tasks:
        tid = str(task_id(task))
        task_record = payload["tasks"].setdefault(
            tid,
            {"name": task["name"], "evidence_types": {}},
        )

        evidence_progress = tqdm(
            hidden_entries(task),
            desc=f"task {tid} factorizations",
            unit="evidence",
            leave=False,
        )
        for entry in evidence_progress:
            evidence_type = int(entry["evidence_type"])
            key = str(evidence_type)
            if resume and key in task_record["evidence_types"]:
                tqdm.write(
                    f"task {tid} evidence {evidence_type}: factorization "
                    "checkpoint already complete"
                )
                continue

            source_text = entry["source_text"]
            evidence_progress.set_postfix(
                evidence=evidence_type,
                alternatives=alternatives,
                stage="generator",
            )
            evidence_progress.refresh()
            tqdm.write(
                f"task {tid} evidence {evidence_type}: "
                f"generating up to {alternatives} factorization alternatives"
            )
            generated, generation_meta = generator_client.generate_json(
                system=GENERATOR_SYSTEM,
                user=factorization_generation_prompt(
                    task,
                    evidence_type,
                    source_text,
                    alternatives,
                    max_components,
                ),
            )

            if not generated.get("factorizable", False):
                record = {
                    "source_text": source_text,
                    "factorizable": False,
                    "non_factorizable_reason": generated.get(
                        "non_factorizable_reason",
                        "Generator found no defensible semantic factorization.",
                    ),
                    "alternatives": [],
                    "generation_metadata": generation_meta,
                    "complete": True,
                }
                task_record["evidence_types"][key] = record
                write_json(output_path, payload, overwrite=True)
                append_jsonl(
                    audit_path,
                    {
                        "kind": "factorization_non_factorizable",
                        "task_id": task_id(task),
                        "evidence_type": evidence_type,
                        "generation_metadata": generation_meta,
                        "reason": record["non_factorizable_reason"],
                    },
                )
                tqdm.write(
                    f"checkpoint: task {tid}, evidence {evidence_type}, "
                    "marked non-factorizable"
                )
                continue

            candidates = [
                normalize_factorization(item, evidence_type, index)
                for index, item in enumerate(
                    generated.get("alternatives", [])[:alternatives]
                )
                if isinstance(item, Mapping)
            ]
            candidates = [
                item
                for item in candidates
                if 2 <= len(item["components"]) <= max_components
            ]

            if not candidates:
                record = {
                    "source_text": source_text,
                    "factorizable": False,
                    "non_factorizable_reason": (
                        "Generator returned no valid multi-component candidate."
                    ),
                    "alternatives": [],
                    "generation_metadata": generation_meta,
                    "complete": True,
                }
                task_record["evidence_types"][key] = record
                write_json(output_path, payload, overwrite=True)
                append_jsonl(
                    audit_path,
                    {
                        "kind": "factorization_invalid_generation",
                        "task_id": task_id(task),
                        "evidence_type": evidence_type,
                        "generation_metadata": generation_meta,
                        "reason": record["non_factorizable_reason"],
                    },
                )
                tqdm.write(
                    f"checkpoint: task {tid}, evidence {evidence_type}, "
                    "no valid factorization alternative"
                )
                continue

            evidence_progress.set_postfix(
                evidence=evidence_type,
                alternatives=len(candidates),
                stage="verifier",
            )
            evidence_progress.refresh()
            verified, verification_meta = verifier_client.generate_json(
                system=VERIFIER_SYSTEM,
                user=factorization_verification_prompt(
                    task,
                    evidence_type,
                    source_text,
                    candidates,
                ),
            )
            verdicts = {
                int(item["candidate_index"]): item
                for item in verified.get("verdicts", [])
                if isinstance(item, Mapping)
                and "candidate_index" in item
            }

            accepted = []
            for index, candidate in enumerate(candidates):
                verdict = verdicts.get(index, {})
                candidate["accepted"] = bool(verdict.get("acceptable", False))
                candidate["quality_score"] = verdict.get("quality_score", 0)
                candidate["verification"] = verdict
                if candidate["accepted"]:
                    accepted.append(candidate)

            selected = (
                max(
                    accepted,
                    key=lambda item: float(item.get("quality_score", 0)),
                )
                if accepted
                else None
            )
            record = {
                "source_text": source_text,
                "factorizable": selected is not None,
                "non_factorizable_reason": (
                    None
                    if selected is not None
                    else "No generated factorization passed verification."
                ),
                "alternatives": candidates,
                "selected_factorization": selected,
                "generation_metadata": generation_meta,
                "verification_metadata": verification_meta,
                "complete": True,
            }
            task_record["evidence_types"][key] = record
            append_jsonl(
                audit_path,
                {
                    "kind": "factorization",
                    "task_id": task_id(task),
                    "evidence_type": evidence_type,
                    "accepted_count": len(accepted),
                    "generation_metadata": generation_meta,
                    "verification_metadata": verification_meta,
                },
            )
            write_json(output_path, payload, overwrite=True)
            tqdm.write(
                f"checkpoint: task {tid}, evidence {evidence_type}, "
                f"factorizations accepted {len(accepted)}/{len(candidates)}"
            )
        evidence_progress.close()


def main() -> None:
    args = parse_args()
    if not 1 <= args.paraphrases_per_type <= 10:
        raise ValidationError("--paraphrases-per-type must be between 1 and 10.")
    if not 1 <= args.factorization_alternatives <= 4:
        raise ValidationError("--factorization-alternatives must be between 1 and 4.")
    if not 2 <= args.max_components <= 4:
        raise ValidationError("--max-components must be between 2 and 4.")
    tasks, _ = extract_tasks_payload(args.input)
    tasks = select_tasks(tasks, args.task_ids)
    paraphrase_generator = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_PARAPHRASE_MODEL",
            default_model="microsoft/gpt-5",
            temperature_env="LLM_PARAPHRASE_TEMPERATURE",
        ),
        progress_callback=tqdm.write,
    )
    factorization_generator = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_FACTORIZATION_MODEL",
            default_model="microsoft/gpt-5.5",
            temperature_env="LLM_FACTORIZATION_TEMPERATURE",
        ),
        progress_callback=tqdm.write,
    )
    paraphrase_verifier = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_PARAPHRASE_VERIFIER_MODEL",
            default_model="microsoft/gpt-5.5",
            temperature_env="LLM_PARAPHRASE_VERIFIER_TEMPERATURE",
        ),
        progress_callback=tqdm.write,
    )
    factorization_verifier = LLMClient(
        LLMConfig.from_env(
            model_env="LLM_FACTORIZATION_VERIFIER_MODEL",
            default_model="microsoft/gpt-5.5",
            temperature_env="LLM_FACTORIZATION_VERIFIER_TEMPERATURE",
        ),
        progress_callback=tqdm.write,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "generation_audit.jsonl"

    # Deliberately keep a task together: a crash after either stage leaves a
    # durable evidence-level checkpoint, and --resume never regenerates an
    # accepted paraphrase or finalized factorization.
    for index, task in enumerate(
        tqdm(tasks, desc="annotating tasks", unit="task"), start=1
    ):
        tid = task_id(task)
        tqdm.write(f"starting task {index}/{len(tasks)} (task_id={tid})")
        continuing = args.resume or index > 1
        if args.mode in {"paraphrases", "both"}:
            generate_paraphrases(
                [task],
                paraphrase_generator,
                paraphrase_verifier,
                args.output_dir / "paraphrases.json",
                audit_path,
                target=args.paraphrases_per_type,
                batch_size=args.paraphrase_batch_size,
                max_rounds=args.max_paraphrase_rounds,
                resume=continuing,
                overwrite=args.overwrite,
                require_complete=args.require_complete,
            )
        if args.mode in {"factorizations", "both"}:
            generate_factorizations(
                [task],
                factorization_generator,
                factorization_verifier,
                args.output_dir / "factorizations.json",
                audit_path,
                alternatives=args.factorization_alternatives,
                max_components=args.max_components,
                resume=continuing,
                overwrite=args.overwrite,
            )
        tqdm.write(f"completed task {index}/{len(tasks)} (task_id={tid})")

    # Mark only a fully traversed run as frozen.  A resumed run may safely
    # replace this metadata after it fills the remaining evidence types.
    for filename in (
        ["paraphrases.json"] if args.mode == "paraphrases"
        else ["factorizations.json"] if args.mode == "factorizations"
        else ["paraphrases.json", "factorizations.json"]
    ):
        path = args.output_dir / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "frozen"
        payload["frozen_after_task_ids"] = [task_id(task) for task in tasks]
        write_json(path, payload, overwrite=True)
        print(f"Wrote frozen {path}")

    print(f"Audit log: {audit_path}")


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
