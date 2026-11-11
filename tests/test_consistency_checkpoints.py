"""Registered checkpoint selection, without GPU inference."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

from pydantic import ValidationError

from segmentation.analysis.metrics import consistency, pipeline, stability
from segmentation.analysis.metrics.protocol import ConsistencyRunInput, StabilityRunInput
from segmentation.catalog import CatalogError
from segmentation.catalog.models import register_model


class DatasetReached(Exception):
    pass


class ConsistencyCheckpointTests(unittest.TestCase):
    runners = (
        (ConsistencyRunInput, consistency.run_consistency),
        (StabilityRunInput, stability.run_stability),
    )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        env = patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(self.root)})
        env.start()
        self.addCleanup(env.stop)
        self.models = []
        for name in ("main", "group"):
            path = self.root / f"{name}.pth"
            path.write_bytes(name.encode())
            self.models.append(
                register_model(
                    root=self.root,
                    name=name,
                    architecture="ScaleProtoSeg",
                    dataset="cityscapes",
                    config={"stage": "push", "grouped": name == "group"},
                    checkpoint=path,
                )
            )

    def test_registration_is_required_before_loading(self):
        with patch.object(pipeline, "load_model_checkpoint") as load:
            with self.assertRaises(CatalogError):
                consistency.run_consistency(ConsistencyRunInput(model="absent", data_type="cityscapes"))
            load.assert_not_called()
        for field in ("checkpoint", "model_name", "training_phase", "group_checkpoint"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ConsistencyRunInput(model="main", data_type="cityscapes", **{field: "raw"})

    def test_explicit_grouping_reference_and_same_model_role_reuse(self):
        for input_type, runner in self.runners:
            for group_reference in (None, "group", self.models[0]["model_id"]):
                with self.subTest(metric=input_type.__name__, group=group_reference):
                    model, group = MagicMock(), MagicMock()
                    model.cuda.return_value = model
                    with (
                        patch.object(pipeline, "load_model_checkpoint", side_effect=[model, group]) as load,
                        patch.object(pipeline, "proto_filter", return_value=[1]) as filtering,
                        patch.object(pipeline, "PatchClassificationDataset", side_effect=DatasetReached),
                    ):
                        with self.assertRaises(DatasetReached):
                            runner(
                                input_type(
                                    model="main",
                                    grouping_model=group_reference,
                                    data_type="cityscapes",
                                )
                            )
                    expected = [call(self.models[0]["checkpoint"]["path"])]
                    if group_reference == "group":
                        expected.append(call(self.models[1]["checkpoint"]["path"]))
                    self.assertEqual(load.call_args_list, expected)
                    if group_reference is None:
                        filtering.assert_not_called()
                    else:
                        filtering.assert_called_once_with(group if group_reference == "group" else model)

    def test_grouping_projection_failures_are_recorded_for_both_metrics(self):
        for input_type, runner in self.runners:
            with self.subTest(metric=input_type.__name__):
                model = MagicMock()
                model.cuda.return_value = model
                with (
                    patch.object(pipeline, "load_model_checkpoint", side_effect=[model, object()]),
                    patch.object(pipeline, "PatchClassificationDataset") as dataset,
                ):
                    with self.assertRaisesRegex(ValueError, "no group projection"):
                        runner(input_type(model="main", grouping_model="group", data_type="cityscapes"))
                dataset.assert_not_called()
        records = [json.loads(path.read_text()) for path in (self.root / "measurements").glob("*/result.json")]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["status"] == "failed" for record in records))
        self.assertTrue(all(record["failure"]["type"] == "ValueError" for record in records))

    def test_changed_weights_fail_before_model_loading(self):
        Path(self.models[0]["checkpoint"]["path"]).write_bytes(b"modified")
        with patch.object(pipeline, "load_model_checkpoint") as load:
            with self.assertRaisesRegex(CatalogError, "changed"):
                consistency.run_consistency(ConsistencyRunInput(model="main", data_type="cityscapes"))
            load.assert_not_called()

    def test_loading_failure_leaves_failed_registered_measurement(self):
        with patch.object(pipeline, "load_model_checkpoint", side_effect=ValueError("bad checkpoint")):
            with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                consistency.run_consistency(ConsistencyRunInput(model="main", data_type="cityscapes"))
        record = json.loads(next((self.root / "measurements").glob("*/result.json")).read_text())
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["model"]["registration_id"], self.models[0]["model_id"])
        self.assertEqual(record["failure"]["type"], "ValueError")
