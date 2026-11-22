"""CPU-only planning and mocked Slurm submission tests; no jobs are launched."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from segmentation.catalog import jobs as jobs
from segmentation.catalog import table as table
from segmentation.catalog.models import model_snapshot
from segmentation.catalog.models import register_model


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
        self.checkpoint.write_bytes(b"prototype")
        self.group.write_bytes(b"group")
        self.main_registration = register_model(
            root=self.root,
            name="model",
            architecture="ScaleProtoSeg",
            dataset="cityscapes",
            config={"stage": "push", "grouped": False},
            checkpoint=self.checkpoint,
        )
        self.group_registration = register_model(
            root=self.root,
            name="group",
            architecture="ScaleProtoSeg",
            dataset="cityscapes",
            config={"stage": "group", "grouped": True},
            checkpoint=self.group,
        )
        self.pascal_checkpoint = self.root / "pascal.pth"
        self.pascal_checkpoint.write_bytes(b"pascal prototype")
        self.pascal_registration = register_model(
            root=self.root,
            name="pascal-model",
            architecture="ScaleProtoSeg",
            dataset="pascal",
            config={"stage": "push", "grouped": False},
            checkpoint=self.pascal_checkpoint,
        )
        environment = patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.record = {
            "measurement_id": "existing",
            "metric": "consistency",
            "status": "complete",
            "dataset": {"name": "cityscapes"},
            "model": {
                "name": "model",
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
        self.record["model"] = model_snapshot(self.main_registration)
        self.record["grouping_model"] = model_snapshot(self.group_registration)
        self.frame = table.records_frame([self.record])
        self.configuration = {field: self.frame.iloc[0][field] for field in table.input_columns(self.frame)}
        self.configuration["parameters.official_parts_only"] = "true"

    def test_planner_infers_exact_checkpoint_and_keeps_hydra_values_quoted(self):
        arguments = jobs.measurement_arguments(self.configuration, self.frame)
        self.assertEqual(arguments[0], "consistency")
        overrides = {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in arguments[1:])}
        self.assertEqual(overrides["model"], self.main_registration["model_id"])
        self.assertEqual(overrides["grouping_model"], self.group_registration["model_id"])
        self.assertIs(overrides["official_parts_only"], True)
        self.assertEqual(overrides["quantile"], 0.8)
        self.assertNotIn("checkpoint_path", overrides)

    def test_planner_rejects_raw_paths_and_unregistered_legacy_inputs(self):
        explicit = {**self.configuration, jobs.PROTOTYPE_PATH: str(self.checkpoint)}
        with self.assertRaisesRegex(ValueError, "cannot be set"):
            jobs.measurement_arguments(explicit, self.frame)
        with self.assertRaisesRegex(ValueError, "registered model"):
            jobs.measurement_arguments(
                {**self.configuration, "model.registration_id": table.MISSING},
                self.frame,
            )

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
            {"model.name": table.MISSING},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                jobs.measurement_arguments(
                    {**self.configuration, **overrides},
                    self.frame,
                )

    def test_stability_uses_its_checkpoint_option_and_does_not_ignore_threshold(self):
        config = {**self.configuration, "metric": "stability"}
        config.update({"parameters.threshold": table.MISSING, "parameters.std_dev": "0.2", "parameters.seed": "7"})
        arguments = jobs.measurement_arguments(config, self.frame)
        self.assertIn(f"model={json.dumps(self.main_registration['model_id'])}", arguments)
        self.assertIn("seed=7", arguments)
        with self.assertRaises(ValueError):
            jobs.measurement_arguments({**config, "parameters.threshold": "0.8"}, self.frame)

    def test_imprecise_activations_is_planned_without_part_parameters(self):
        config = {**self.configuration, "metric": "imprecise_activations"}
        input_type = jobs.RUN_INPUTS["imprecise_activations"]
        for field in config:
            if field.startswith("parameters.") and field[len("parameters.") :] not in input_type.model_fields:
                config[field] = table.MISSING

        arguments = jobs.measurement_arguments(config, self.frame)
        overrides = {key: json.loads(value) for key, value in (arg.split("=", 1) for arg in arguments[1:])}
        self.assertEqual(arguments[0], "imprecise_activations")
        self.assertEqual(overrides["model"], self.main_registration["model_id"])
        self.assertEqual(overrides["grouping_model"], self.group_registration["model_id"])
        self.assertEqual(overrides["data_type"], "cityscapes")
        self.assertNotIn("quantile", overrides)
        self.assertNotIn("official_parts_only", overrides)

    def test_registration_drives_dataset_without_cross_dataset_combinations(self):
        configurations, errors = jobs.plan_configurations(
            "imprecise_activations",
            [self.main_registration["model_id"], self.pascal_registration["model_id"]],
            {},
            {},
            self.frame,
            root=self.root,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(configurations), 2)
        datasets = {
            config["model.registration_id"]: config["dataset.name"] for config in configurations
        }
        self.assertEqual(
            datasets,
            {
                self.main_registration["model_id"]: "cityscapes",
                self.pascal_registration["model_id"]: "pascal",
            },
        )

    def test_metric_parameters_form_only_the_selected_product(self):
        configurations, errors = jobs.plan_configurations(
            "consistency",
            [self.main_registration["model_id"]],
            {},
            {"quantile": ["0.7", "0.8"], "threshold": ["0.6", "0.9"]},
            self.frame,
            root=self.root,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(configurations), 4)
        self.assertEqual(
            {
                (config["parameters.quantile"], config["parameters.threshold"])
                for config in configurations
            },
            {("0.7", "0.6"), ("0.7", "0.9"), ("0.8", "0.6"), ("0.8", "0.9")},
        )

    def test_grouping_registration_must_match_each_models_dataset(self):
        with self.assertRaisesRegex(ValueError, "same dataset"):
            jobs.plan_configurations(
                "imprecise_activations",
                [self.main_registration["model_id"]],
                {"cityscapes": self.pascal_registration["model_id"]},
                {},
                self.frame,
                root=self.root,
            )

    def test_empty_catalog_can_seed_a_plan_and_matching_is_exact(self):
        empty = table.records_frame([])
        configurations, errors = jobs.plan_configurations(
            "imprecise_activations",
            [self.main_registration["model_id"]],
            {},
            {},
            empty,
            root=self.root,
        )
        self.assertEqual(errors, [])
        self.assertEqual(len(configurations), 1)
        self.assertTrue(jobs.matching_measurements(empty, configurations[0]).empty)
        exact = {field: self.frame.iloc[0][field] for field in table.input_columns(self.frame)}
        measured = jobs.matching_measurements(self.frame, exact)
        self.assertEqual(len(measured), 1)
        changed = {**exact, "parameters.threshold": "0.7"}
        self.assertTrue(jobs.matching_measurements(self.frame, changed).empty)

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

    def _write_receipt(self, *, status="submitted", job_id="12345"):
        arguments = jobs.measurement_arguments(self.configuration, self.frame, root=self.root)
        path = jobs.submission_path(self.root, arguments)
        path.parent.mkdir(exist_ok=True)
        receipt = {"status": status, "job_id": job_id, "arguments": arguments}
        path.write_text(json.dumps(receipt))
        return path, receipt

    def test_retry_archives_terminal_receipt_and_allows_new_submission(self):
        path, receipt = self._write_receipt()
        queue = subprocess.CompletedProcess([], 0, stdout="")
        accounting = subprocess.CompletedProcess([], 0, stdout="12345|FAILED\n")
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs.subprocess, "run", side_effect=[accounting, queue]
        ):
            result = jobs.release_submission_for_retry(self.configuration, self.root)
        self.assertEqual((result["job_id"], result["state"]), ("12345", "FAILED"))
        self.assertFalse(path.exists())
        self.assertEqual(json.loads(Path(result["receipt"]).read_text()), receipt)
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs, "validate_runtime", return_value=self.root / "python"
        ), patch.object(jobs, "current_slurm_node", return_value="maxitank"), patch.object(
            jobs.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="12346\n")
        ):
            self.assertEqual(
                jobs.submit_measurement(self.configuration, self.root, self.root, "python")["job_id"],
                "12346",
            )
        self.assertEqual(json.loads(Path(result["receipt"]).read_text()), receipt)

    def test_retry_blocks_running_missing_and_ambiguous_slurm_records(self):
        cases = [
            ("12345|RUNNING\n", "12345|FAILED\n", "still RUNNING"),
            ("", "12345|PENDING\n", "PENDING"),
            ("", "", "unambiguous"),
            ("", "12345|FAILED\n12345|FAILED\n", "unambiguous"),
            ("", "12345|\n", "No accounting state"),
        ]
        for queue, accounting, expected in cases:
            with self.subTest(queue=queue, accounting=accounting):
                path, _ = self._write_receipt()
                responses = [
                    subprocess.CompletedProcess([], 0, stdout=accounting),
                    subprocess.CompletedProcess([], 0, stdout=queue),
                ]
                with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
                    jobs.subprocess, "run", side_effect=responses
                ), self.assertRaisesRegex(ValueError, expected):
                    jobs.release_submission_for_retry(self.configuration, self.root)
                self.assertTrue(path.exists())

    def test_retry_keeps_uncertain_receipt_and_rechecks_catalog(self):
        path, _ = self._write_receipt(status="submission_error", job_id=None)
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs.subprocess, "run"
        ) as run:
            with self.assertRaisesRegex(ValueError, "uncertain"):
                jobs.release_submission_for_retry(self.configuration, self.root)
            run.assert_not_called()
        current = {**self.record, "parameters": {**self.record["parameters"], "official_parts_only": True}}
        with patch.object(jobs.table, "load_records", return_value=([current], [])):
            with self.assertRaisesRegex(ValueError, "catalog entry"):
                jobs.release_submission_for_retry(self.configuration, self.root)
        self.assertTrue(path.exists())

    def test_retry_keeps_receipt_when_slurm_is_unavailable(self):
        path, _ = self._write_receipt()
        with patch.object(jobs.table, "load_records", return_value=([self.record], [])), patch.object(
            jobs.subprocess, "run", side_effect=subprocess.TimeoutExpired("squeue", 30)
        ), self.assertRaises(subprocess.TimeoutExpired):
            jobs.release_submission_for_retry(self.configuration, self.root)
        self.assertTrue(path.exists())

    def test_retry_rechecks_catalog_after_slurm_lookup(self):
        path, _ = self._write_receipt()
        current = {**self.record, "parameters": {**self.record["parameters"], "official_parts_only": True}}
        with patch.object(
            jobs.table, "load_records", side_effect=[([self.record], []), ([self.record], []), ([current], [])]
        ), patch.object(jobs, "terminal_job_state", return_value="FAILED"):
            with self.assertRaisesRegex(ValueError, "catalog entry"):
                jobs.release_submission_for_retry(self.configuration, self.root)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
