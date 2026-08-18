from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator import generate_dataset_in_memory, generate_task, write_dataset
from validation import validate_dataset_directory, validate_task


class GeneratorTests(unittest.TestCase):
    def test_reasoning_depths_1_to_4(self) -> None:
        for depth in (1, 2, 3, 4):
            with self.subTest(depth=depth):
                task = generate_task(
                    task_id=f"depth_{depth}",
                    task_seed=1000 + depth,
                    dataset_seed=123,
                    task_index=depth,
                    population_size=12,
                    reasoning_depth=depth,
                    support_redundancy=3,
                    distractors=2,
                    distractor_redundancy=1,
                    num_options=3,
                    no_single_agent_solution=False,
                )
                self.assertEqual([], validate_task(task))
                self.assertEqual(depth, len(task["query"]["supporting_fact_ids"]))

    def test_no_single_agent_solution(self) -> None:
        task = generate_task(
            task_id="hidden_profile_like",
            task_seed=999,
            dataset_seed=42,
            task_index=1,
            population_size=24,
            reasoning_depth=2,
            support_redundancy=6,
            distractors=4,
            distractor_redundancy=1,
            num_options=3,
            no_single_agent_solution=True,
        )
        support = set(task["query"]["supporting_fact_ids"])
        self.assertTrue(support)
        for payload in task["agents"].values():
            self.assertFalse(support.issubset(set(payload["fact_ids"])))
        self.assertEqual([], validate_task(task))

    def test_depth_one_no_single_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_task(
                task_id="impossible",
                task_seed=1,
                dataset_seed=1,
                task_index=1,
                population_size=8,
                reasoning_depth=1,
                support_redundancy=2,
                distractors=0,
                num_options=3,
                no_single_agent_solution=True,
            )

    def test_same_seed_same_dataset_in_memory(self) -> None:
        kwargs = dict(
            num_tasks=8,
            population_size=16,
            reasoning_depth=3,
            support_redundancy=4,
            distractors=3,
            distractor_redundancy=2,
            num_options=4,
            seed=314159,
            no_single_agent_solution=True,
        )
        first = generate_dataset_in_memory(**kwargs)
        second = generate_dataset_in_memory(**kwargs)
        self.assertEqual(first, second)

    def test_written_dataset_reproducible_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            one = base / "one"
            two = base / "two"
            kwargs = dict(
                num_tasks=5,
                population_size=12,
                reasoning_depth=2,
                support_redundancy=4,
                distractors=2,
                distractor_redundancy=1,
                num_options=3,
                seed=77,
                no_single_agent_solution=True,
            )
            write_dataset(output_dir=one, **kwargs)
            write_dataset(output_dir=two, **kwargs)
            self.assertEqual([], validate_dataset_directory(one, check_reproducibility=True))
            self.assertEqual([], validate_dataset_directory(two, check_reproducibility=True))

            files_one = sorted(p.name for p in one.iterdir())
            files_two = sorted(p.name for p in two.iterdir())
            self.assertEqual(files_one, files_two)
            for name in files_one:
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())

    def test_validator_detects_agent_unknown_fact(self) -> None:
        task = generate_task(
            task_id="corrupt_me",
            task_seed=5,
            dataset_seed=5,
            task_index=1,
            population_size=10,
            reasoning_depth=2,
            support_redundancy=3,
            distractors=1,
            distractor_redundancy=1,
            num_options=3,
            no_single_agent_solution=True,
        )
        first_agent = next(iter(task["agents"]))
        task["agents"][first_agent]["fact_ids"].append("does_not_exist")
        errors = validate_task(task)
        self.assertTrue(any("unknown facts" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
