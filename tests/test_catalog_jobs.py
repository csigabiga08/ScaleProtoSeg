"""CPU-only planning and mocked Slurm submission tests; no jobs are launched."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from segmentation.catalog import jobs as jobs
from segmentation.catalog import table as table


class CatalogJobsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "measurements").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts/run_metric.slurm").touch()
        self.checkpoint = self.root / "prototype with spaces.pth"
        self.group = self.root / "group.pth"
        self.checkpoint.touch()
        self.group.touch()
        self.record = {
            "measurement_id": "existing",
            "metric": "consistency",
            "status": "complete",
            "dataset": {"name": "cityscapes"},
            "model": {
                "label": "model",
                "training_stage": "push",
                "inputs": {
                    "prototype_checkpoint": {"path": str(self.checkpoint)},
                    "group_checkpoint": {"path": str(self.group)},
                },
            },
            "parameters": {
                "quantile": 0.8,
                "threshold": 0.8,
                "official_parts_only": False,
                "require_semantic_match": False,
                "exclude_background_centroid": False,
                "merge_left_right_parts": False,
            },
            "scores": {"consistency": {"value": 0.5}},
        }
        self.frame = table.records_frame([self.record])
        self.configuration = {field: self.frame.iloc[0][field] for field in table.input_columns(self.frame)}
        self.configuration["parameters.official_parts_only"] = "true"

    def test_planner_infers_exact_checkpoint_and_keeps_hydra_values_quoted(self):
        arguments = jobs.measurement_arguments(self.configuration, self.frame)
        self.assertEqual(arguments[0], "consistency")
        overrides = {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in arguments[1:])}
        self.assertEqual(overrides["checkpoint"], str(self.checkpoint))
        self.assertEqual(overrides["group_checkpoint"], str(self.group))
        self.assertIs(overrides["official_parts_only"], True)
        self.assertEqual(overrides["quantile"], 0.8)
        self.assertNotIn("checkpoint_path", overrides)

    def test_planner_rejects_ambiguous_checkpoint_and_accepts_explicit_path(self):
        other = self.frame.copy()
        other[jobs.PROTOTYPE_PATH] = str(self.group)
        ambiguous = pd.concat([self.frame, other])
        with self.assertRaisesRegex(ValueError, "No unique prototype checkpoint"):
            jobs.measurement_arguments(self.configuration, ambiguous)
        explicit = {**self.configuration, jobs.PROTOTYPE_PATH: str(self.checkpoint)}
        self.assertIn(f"checkpoint={json.dumps(str(self.checkpoint))}", jobs.measurement_arguments(explicit, ambiguous))
        with self.assertRaisesRegex(ValueError, "Checkpoint does not exist"):
            jobs.measurement_arguments({**explicit, jobs.PROTOTYPE_PATH: str(self.root / "absent")}, self.frame)

    def test_planner_rejects_invalid_or_unrepresentable_configurations(self):
        invalid = (
            {"dataset.name": "pascal"},
            {"parameters.quantile": "nan"},
            {"parameters.quantile": "2"},
            {"parameters.quantile": "0.80"},
            {"parameters.official_parts_only": table.MISSING},
            {"parameters.official_parts_only": "yes"},
            {"execution.commit": "other"},
            {"parameters.seed": "1"},
            {"model.label": table.MISSING},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                jobs.measurement_arguments(
                    {**self.configuration, jobs.PROTOTYPE_PATH: str(self.checkpoint), **overrides},
                    self.frame,
                )

    def test_stability_uses_its_checkpoint_option_and_does_not_ignore_threshold(self):
        config = {**self.configuration, "metric": "stability", jobs.PROTOTYPE_PATH: str(self.checkpoint)}
        config.update({"parameters.threshold": table.MISSING, "parameters.std_dev": "0.2", "parameters.seed": "7"})
        arguments = jobs.measurement_arguments(config, self.frame)
        self.assertIn(f"checkpoint_path={json.dumps(str(self.checkpoint))}", arguments)
        self.assertIn("seed=7", arguments)
        with self.assertRaises(ValueError):
            jobs.measurement_arguments({**config, "parameters.threshold": "0.8"}, self.frame)

    def test_runtime_keeps_virtualenv_interpreter_path(self):
        binary = self.root / "base-python"
        binary.touch(mode=0o755)
        interpreter = self.root / "venv-python"
        interpreter.symlink_to(binary)
        with patch.object(jobs.subprocess, "run") as run:
            self.assertEqual(jobs.validate_runtime(interpreter), interpreter)
            self.assertEqual(run.call_args.args[0][0], str(interpreter))

    def test_node_is_discovered_from_the_webapp_host_not_a_user_override(self):
        with patch.object(jobs.socket, "gethostname", return_value="maxitank.example"):
            with patch.object(
                jobs.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [],
                    0,
                    stdout="NodeName=maxitank NodeHostName=maxitank",
                ),
            ) as run:
                self.assertEqual(jobs.current_slurm_node(), "maxitank")
                self.assertEqual(run.call_args.args[0], ["scontrol", "show", "node", "maxitank", "-o"])

    def test_submission_receipt_is_persistent_and_prevents_duplicate_jobs(self):
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs, "validate_runtime", return_value=self.root / "python"
        ), patch.object(jobs, "current_slurm_node", return_value="maxitank"), patch.object(
            jobs.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                [],
                0,
                stdout="12345;cluster\n",
            ),
        ) as run:
            receipt = jobs.submit_measurement(self.configuration, self.root, self.root, self.root / "python")
            self.assertEqual(receipt["job_id"], "12345")
            command = run.call_args.args[0]
            self.assertIn("--nodelist=maxitank", command)
            self.assertEqual(command[0:2], ["sbatch", "--parsable"])
            self.assertEqual(run.call_args.kwargs["env"]["MEASUREMENT_CATALOG_ROOT"], str(self.root))
            self.assertEqual(run.call_args.kwargs["env"]["METRIC_PYTHON"], str(self.root / "python"))
            self.assertEqual(jobs.submission_record(self.root, receipt["arguments"])["job_id"], "12345")
            with self.assertRaisesRegex(ValueError, "already has a submission receipt"):
                jobs.submit_measurement(self.configuration, self.root, self.root, self.root / "python")
            self.assertEqual(run.call_count, 1)

    def test_submission_rechecks_catalog_before_launching(self):
        current = {**self.record, "parameters": {**self.record["parameters"], "official_parts_only": True}}
        with patch.object(jobs.table, "load_records", return_value=([current], [])):
            with patch.object(jobs.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "now has a catalog entry"):
                    jobs.submit_measurement(self.configuration, self.root, self.root, "python")
                run.assert_not_called()
        with patch.object(jobs.table, "load_records", return_value=([], [("bad", "unreadable")])):
            with self.assertRaisesRegex(ValueError, "Unreadable"):
                jobs.submit_measurement(self.configuration, self.root, self.root, "python")

    def test_submission_timeout_keeps_a_receipt_to_prevent_unsafe_retry(self):
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs, "validate_runtime", return_value=self.root / "python"
        ), patch.object(jobs, "current_slurm_node", return_value="maxitank"), patch.object(
            jobs.subprocess, "run", side_effect=subprocess.TimeoutExpired("sbatch", 30)
        ) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                jobs.submit_measurement(self.configuration, self.root, self.root, "python")
            arguments = jobs.measurement_arguments(self.configuration, self.frame)
            self.assertEqual(jobs.submission_record(self.root, arguments)["status"], "submission_error")
            with self.assertRaisesRegex(ValueError, "already has a submission receipt"):
                jobs.submit_measurement(self.configuration, self.root, self.root, "python")
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
