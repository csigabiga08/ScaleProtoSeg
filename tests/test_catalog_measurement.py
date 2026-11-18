"""Exercise catalog detail navigation and analysis in Streamlit's test runner."""

import copy
import csv
import hashlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


APP = Path(__file__).resolve().parents[1] / "segmentation/catalog/app.py"


@unittest.skipIf(AppTest is None, "Install requirements/requirements-catalog.txt for Streamlit UI tests")
class CatalogMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bundle = root / "measurements" / "fixture"
        self.bundle.mkdir(parents=True)
        content = (
            "sample_id,class_id,prototype_id,repetition,part_0,part_1,part_2,part_3\n"
            "a,6,1,0,,1,0,\n"
            "b,6,1,0,,,0,1\n"
            "a,6,2,0,,0,1,0\n"
            "a,6,3,0,,0,0,0\n"
        ).encode()
        (self.bundle / "observations.csv").write_bytes(content)
        self.record = {
            "measurement_id": "fixture",
            "metric": "consistency",
            "status": "complete",
            "model": {"label": "fixture", "inputs": {}},
            "dataset": {"name": "pascal"},
            "parameters": {"quantile": 0.8},
            "protocol": {
                "observation_columns": [f"part_{pid}" for pid in range(4)],
                "classes": {"evaluation_class_id_to_label": {"6": "bottle"}},
                "consistency": {"threshold": 0.8},
            },
            "scores": {"consistency": {"value": 2 / 3, "numerator": 2, "denominator": 3, "unit": "fraction"}},
            "artifacts": {
                "observations": {
                    "path": "observations.csv",
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            },
        }
        self.write_record()
        environment = patch.dict(os.environ, {"MEASUREMENT_CATALOG_ROOT": str(root)})
        environment.start()
        self.addCleanup(environment.stop)
        import streamlit as st

        st.cache_data.clear()

    def write_record(self):
        (self.bundle / "result.json").write_text(json.dumps(self.record))

    def detail(self, measurement_id="fixture"):
        app = AppTest.from_file(str(APP), default_timeout=20)
        app.query_params["measurement"] = measurement_id
        return app.run()

    def test_catalog_links_detail_analysis_filters_and_back_navigation(self):
        import streamlit as st

        with patch("streamlit.download_button", wraps=st.download_button) as download:
            app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        runs = next(item for item in app.dataframe if item.key == "runs")
        self.assertNotIn(("", "measurement_page"), runs.value.columns)
        target = runs.value.loc["fixture", ("", "measurement_id")]
        self.assertEqual(target, "?measurement=fixture")
        position = runs.value.index.nlevels + runs.value.columns.get_loc(("", "measurement_id"))
        config = json.loads(runs.proto.columns)[f"_pos:{position}"]["type_config"]
        self.assertEqual(config["type"], "link")
        self.assertEqual(re.search(config["display_text"], target).group(1), "fixture")
        exported = next(call.args[1] for call in download.call_args_list if call.kwargs.get("key") == "runs_csv")
        row = next(csv.DictReader(io.StringIO(exported)))
        self.assertEqual(row["measurement_id"], "fixture")
        self.assertNotIn("measurement_page", row)
        app.query_params["measurement"] = parse_qs(urlsplit(target).query)["measurement"][0]
        app.run()
        self.assertFalse(app.exception)
        self.assertIn("fixture", app.title[0].value)
        app.checkbox(key="analyze_fixture").check().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.dataframe[1].value.shape, (3, 3))
        self.assertAlmostEqual(app.dataframe[1].value.loc[1, 2], -1)
        app.checkbox[1].check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[1].value.shape, (2, 2))
        app.checkbox[2].check().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[2].value.loc[1, (6, "part_1")], 1)
        self.assertTrue(app.dataframe[2].value[(6, "part_3")].isna().all())
        app.number_input[0].set_value(1.0).run()
        self.assertFalse(app.exception)
        self.assertIn("No prototypes match", app.info[0].value)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertNotIn("measurement", app.query_params)
        self.assertEqual(app.title[0].value, "Measurement catalog")

    def test_catalog_groups_sidebar_filters_and_table_headers(self):
        self.record["model"]["training_stage"] = "push"
        self.record["parameters"]["flag"] = False
        self.write_record()
        second = copy.deepcopy(self.record)
        second["measurement_id"] = "second"
        second["model"].update(label="other", training_stage="prune")
        second["parameters"]["flag"] = True
        bundle = self.bundle.parent / "second"
        bundle.mkdir()
        (bundle / "result.json").write_text(json.dumps(second))

        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        sections = {section.label: section for section in app.sidebar.expander}
        self.assertIn("model", sections)
        self.assertEqual(
            [widget.label for widget in sections["model"].multiselect],
            ["label", "training_stage"],
        )
        self.assertEqual([widget.label for widget in sections["parameters"].multiselect], ["flag"])
        runs = next(item for item in app.dataframe if item.key == "runs")
        self.assertIn(("model", "label"), runs.value.columns)
        self.assertIn(("model", "training_stage"), runs.value.columns)
        self.assertEqual(len(runs.value), 2)
        config = json.loads(runs.proto.columns)
        link_position = runs.value.index.nlevels + runs.value.columns.get_loc(("", "measurement_id"))
        self.assertEqual(config[f"_pos:{link_position}"]["type_config"]["type"], "link")
        self.assertEqual(config["_pos:1"]["type_config"]["format"], "%.4f")
        self.assertNotIn("_pos:0", config)
        self.assertTrue(any(("score", "Off") in item.value for item in app.dataframe))

        app.multiselect(key="coverage_columns").set_value(["parameters.flag"]).run()
        self.assertFalse(app.exception)
        coverage = next(item.value for item in app.dataframe if ("score", "parameters.flag=false") in item.value)
        self.assertIn(("model", "label"), coverage.columns)
        missing = next(item for item in app.dataframe if (item.key or "").startswith("coverage_missing_"))
        self.assertIn(("model", "training_stage"), missing.value.columns)

        app.multiselect(key="filter_model.label").set_value(["fixture"]).run()
        self.assertFalse(app.exception)
        runs = next(item.value for item in app.dataframe if item.key == "runs")
        self.assertEqual(list(runs.index), ["fixture"])
        self.assertEqual(app.multiselect(key="filter_model.label").value, ["fixture"])

    def test_unknown_and_running_measurements_and_bad_hash(self):
        app = self.detail("missing")
        self.assertFalse(app.exception)
        self.assertIn("not found", app.error[0].value)
        self.record["status"] = "running"
        self.write_record()
        app = self.detail()
        self.assertFalse(app.exception)
        self.assertIn("no finalized", app.info[0].value)
        self.assertEqual(len(app.checkbox), 0)
        self.record["status"] = "complete"
        self.record["artifacts"]["observations"]["sha256"] = "0" * 64
        self.write_record()
        app = self.detail()
        app.checkbox(key="analyze_fixture").check().run()
        self.assertFalse(app.exception)
        self.assertIn("hash mismatch", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
