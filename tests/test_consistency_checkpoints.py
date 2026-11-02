"""Checkpoint selection through the consistency CLI, without GPU inference."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import argh

from segmentation.analysis.metrics import consistency


class DatasetReached(Exception):
    pass


class ConsistencyCheckpointTests(unittest.TestCase):
    def test_checkpoint_selection_through_cli(self):
        cases = [
            ([], False, ["model/checkpoints/push_last.pth"], None),
            (["--group-name", "group"], False,
             ["model/checkpoints/push_last.pth", "group/checkpoints/th-0.05-nopush-group_last.pth"], "group"),
            (["--checkpoint", "/custom/best.pth"], False, ["/custom/best.pth"], None),
            (["--checkpoint", "/custom/best.pth"], True, ["/custom/best.pth"], "model"),
            (["--group-name", "group", "--group-checkpoint", "/custom/group.pth"], False,
             ["model/checkpoints/push_last.pth", "/custom/group.pth"], "group"),
            (["--checkpoint", "/custom/best.pth", "--group-checkpoint", "/custom/group.pth"], True,
             ["/custom/best.pth", "/custom/group.pth"], "group"),
        ]
        for flags, grouped, paths, filter_source in cases:
            with self.subTest(flags=flags, grouped=grouped), tempfile.TemporaryDirectory() as root:
                model = MagicMock()
                model.cuda.return_value = model
                if not grouped:
                    del model.group_projection
                group = MagicMock()
                with (
                    patch.dict(os.environ, {"RESULTS_DIR": root}),
                    patch.object(consistency, "load_model_checkpoint", side_effect=[model, group]) as load,
                    patch.object(consistency, "proto_filter", return_value=[1]) as proto_filter,
                    patch.object(consistency, "PatchClassificationDataset", side_effect=DatasetReached),
                ):
                    with self.assertRaises(DatasetReached):
                        argh.dispatch_command(
                            consistency.run_consistency, argv=["model", "push", "cityscapes", *flags]
                        )
                    self.assertEqual(load.call_args_list, [call(os.path.join(root, p)) for p in paths])
                    if filter_source is None:
                        proto_filter.assert_not_called()
                    else:
                        proto_filter.assert_called_once_with(model if filter_source == "model" else group)


if __name__ == "__main__":
    unittest.main()
