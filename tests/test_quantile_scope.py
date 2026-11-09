"""Whole-image quantiles and metric CLI integration, without GPU inference."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import argh
import numpy as np
import torch
from PIL import Image

from segmentation.analysis.equivariance import quantile_map, quantile_map_whole_image
from segmentation.analysis.metrics import consistency, stability


class WholeImageQuantileTests(unittest.TestCase):
    def test_image_quantile_uses_all_pixels_and_activates_outside_class(self):
        values = torch.tensor([[[1., 2., 3., 4.]], [[4., 3., 2., 1.]]])
        mask = torch.tensor([[True, True, False, False]])
        result = quantile_map(values, 0.5, mask, scope="image")
        expected = torch.tensor([[[False, True], [False, True], [True, False], [True, False]]])
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(result, quantile_map_whole_image(values, 0.5)))
        self.assertEqual(result.dtype, torch.bool)
        self.assertEqual(result.device, values.device)
        self.assertEqual(result.shape, (1, 4, 2))
        class_expected = torch.tensor([[[False, True], [True, False], [False, False], [False, False]]])
        self.assertTrue(torch.equal(quantile_map(values, 0.5, mask), class_expected))
        self.assertTrue(torch.equal(quantile_map(values, 0.5, mask, scope="class"), class_expected))

    def test_image_mode_ignores_empty_class_mask(self):
        values = torch.tensor([[[1., 2., 3.]]])
        mask = torch.zeros((1, 3), dtype=torch.bool)
        self.assertFalse(quantile_map(values, 0.5, mask).any())
        self.assertTrue(torch.equal(quantile_map(values, 0.5, mask, scope="image"),
                                    torch.tensor([[[False], [False], [True]]])))

    def test_image_includes_zeros_and_preserves_ties_endpoints_and_low_precision(self):
        values = torch.tensor([[[0., 0., 0., 1., 2.]]], dtype=torch.float16)
        self.assertTrue(torch.equal(quantile_map_whole_image(values, 0.5),
                                    torch.tensor([[[False], [False], [False], [True], [True]]])))
        self.assertFalse(quantile_map_whole_image(torch.ones((1, 2, 3)), 0.8).any())
        self.assertFalse(quantile_map_whole_image(values, 1.).any())
        self.assertTrue(torch.equal(quantile_map_whole_image(values, 0.),
                                    torch.tensor([[[False], [False], [False], [True], [True]]])))

    def test_global_nonfinite_pixels_are_rejected(self):
        for outside in (float('nan'), float('inf'), float('-inf')):
            values = torch.tensor([[[1., 2., outside]]])
            mask = torch.tensor([[True, True, False]])
            quantile_map(values, 0.5, mask)  # Class mode never selects outside.
            with self.assertRaises(ValueError):
                quantile_map(values, 0.5, mask, scope="image")

    def test_invalid_scope_missing_class_mask_and_bad_quantile_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scope'):
            quantile_map(torch.ones((1, 2, 3)), 0.5, scope="unknown")
        with self.assertRaisesRegex(ValueError, 'class_mask is required'):
            quantile_map(torch.ones((1, 2, 3)), 0.5)
        for q in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                quantile_map_whole_image(torch.ones((1, 2, 3)), q)


class MetricScopeTests(unittest.TestCase):
    def test_real_centroid_hits_change_with_scope_in_both_metrics(self):
        image = Image.new("RGB", (3, 3))
        semantic = np.zeros((3, 3), dtype=np.int64)
        semantic[0, :2] = 2
        parts = np.zeros((3, 3), dtype=np.int64)
        parts[0, 1] = 1
        distances = torch.full((1, 1, 3, 3), 0.1)  # Strong outside-class activations.
        distances[0, 0, 0, 0] = 10.
        distances[0, 0, 0, 1] = 5.
        model = MagicMock()
        model.epsilon = 1e-4
        model.prototype_class_identity = torch.zeros((1, 21))
        model.prototype_class_identity[0, 1] = 1
        model.forward_from_conv_features.return_value = (None, distances)
        for metric in (consistency, stability):
            with (
                self.subTest(metric=metric.__name__),
                patch.object(metric, 'to_normalized_tensor', return_value=torch.zeros((3, 3, 3))),
                patch.object(torch.Tensor, 'cuda', lambda self: self),
                patch.object(metric, 'MAX_PARTS', 24, create=True),
            ):
                extra_options = [{}, {'noise': 0.2}] if metric is stability else [{}]
                for extra in extra_options:
                    _, class_hits = metric.part_intersect(
                        image, semantic, parts, model, {1: 'aeroplane'}, 'sample', 0.5,
                        exclude_background_centroid=True, **extra,
                    )
                    _, image_hits = metric.part_intersect(
                        image, semantic, parts, model, {1: 'aeroplane'}, 'sample', 0.5,
                        exclude_background_centroid=True, quantile_scope='image', **extra,
                    )
                    self.assertEqual(class_hits[0][1], 1)
                    self.assertEqual(image_hits[0][1], 0)

    def test_cli_scope_reaches_all_passes_and_files_do_not_overwrite(self):
        for metric in (consistency, stability):
            with self.subTest(metric=metric.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for folder in ('annotations', 'annotations_PIDS', 'annotations_SIDS'):
                    d = root / folder / 'val'
                    d.mkdir(parents=True)
                    np.save(d / 'sample.npy', [[1]])
                image = root / 'sample.png'
                Image.new('RGB', (1, 1)).save(image)
                dataset = MagicMock()
                dataset.__len__.return_value = 1
                dataset.img_ids = ['sample']
                dataset.annotations_dir = str(root / 'annotations/val')
                dataset.get_img_path.return_value = str(image)
                dataset.convert_targets.side_effect = lambda a: a + 1
                model = MagicMock()
                model.cuda.return_value = model
                del model.group_projection
                observation = [0] * (metric.MAX_PARTS_PASCAL + 1) + [0, 'aeroplane', 'sample']
                name = metric.__name__.split('.')[-1]
                checkpoint_flag = '--checkpoint' if name == 'consistency' else '--checkpoint-path'
                checkpoint = root / 'model/checkpoints/push_last.pth'
                argv = ['model', 'push', 'pascal', checkpoint_flag, str(checkpoint),
                        '--require-semantic-match', '--exclude-background-centroid', '--merge-left-right-parts']
                with (
                    patch.dict(os.environ, {'RESULTS_DIR': str(root)}),
                    patch.object(metric, 'load_model_checkpoint', return_value=model),
                    patch.object(metric, 'PatchClassificationDataset', return_value=dataset),
                    patch.object(metric, 'part_intersect', return_value=({}, [observation])) as intersect,
                ):
                    command = getattr(metric, 'run_' + name)
                    argh.dispatch_command(command, argv=argv)
                    out = root / 'model/metrics-filteredparts-foreground'
                    original = {p.name: p.read_bytes() for p in out.iterdir()}
                    argh.dispatch_command(command, argv=argv + ['--quantile-scope', 'image'])
                    for filename, content in original.items():
                        self.assertEqual((out / filename).read_bytes(), content)
                        self.assertNotIn('qscope_image', filename)
                    factor = 1 if name == 'consistency' else 2
                    self.assertEqual([c.kwargs['quantile_scope'] for c in intersect.call_args_list],
                                     ['class'] * factor + ['image'] * factor)
                    for c in intersect.call_args_list:
                        self.assertTrue(c.kwargs['merge_left_right_parts'])
                        self.assertTrue(c.kwargs['exclude_background_centroid'])
                    descriptions = sorted(out.glob('description_*.json'))
                    self.assertEqual(len(descriptions), 2)
                    self.assertEqual({json.loads(p.read_text())['parameters']['quantile_scope']
                                      for p in descriptions}, {'class', 'image'})
                    for p in descriptions:
                        data = json.loads(p.read_text())
                        self.assertEqual(data['status'], 'complete')
                    self.assertEqual(len(list(out.glob(name + '_score_*qscope_image*.txt'))), 1)
                    self.assertEqual(len(list(out.glob('*qscope_image*.csv'))),
                                     2 if name == 'consistency' else 1)

    def test_invalid_scope_rejected_before_loading_model(self):
        for metric in (consistency, stability):
            command = getattr(metric, 'run_' + metric.__name__.split('.')[-1])
            with patch.object(metric, 'load_model_checkpoint') as load:
                with self.assertRaisesRegex(ValueError, 'quantile_scope'):
                    command('model', 'push', 'pascal', quantile_scope='invalid')
                load.assert_not_called()
