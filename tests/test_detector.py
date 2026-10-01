from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from mot_data import load_config
from test_mot import fixture
import detector


class CacheTests(unittest.TestCase):
    def test_code_hash_portable_across_line_endings(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root)/'a.py', Path(root)/'b.py'
            a.write_bytes(b'x = 1\ny = 2\n')
            b.write_bytes(b'x = 1\r\ny = 2\r\n')
            self.assertEqual(detector.code_hash(a), detector.code_hash(b))

    def test_cache_lock_released_after_error(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                with detector.cache_lock(root):
                    with self.assertRaises(RuntimeError):
                        with detector.cache_lock(root):
                            pass
                    raise ValueError('fixture')
            with detector.cache_lock(root):
                pass

    def test_resume_hash_validation_and_no_skipped_empty_frame(self):
        with tempfile.TemporaryDirectory() as root:
            sequence = fixture(root)
            cache = Path(root)/'cache'
            config = load_config()
            versions = {'torch':'fixture', 'torchvision':'fixture'}
            def fake(model, transform, path, frame, device):
                return np.array([[frame,10,10,10,10,.9]]) if frame != 2 else np.empty((0,6))
            with mock.patch.object(detector, 'own_nms_context', return_value=nullcontext()), mock.patch.object(detector, 'infer_frame', side_effect=fake) as infer:
                detector.extract_detections(sequence, config, cache, None, None, 'cpu', versions, max_frames=1)
                self.assertEqual(infer.call_count, 1)
                with self.assertRaises(ValueError):
                    detector.load_cached_detections(sequence, config, cache)
                first = detector.extract_detections(sequence, config, cache, None, None, 'cpu', versions)
                self.assertEqual(infer.call_count, 3)
                second = detector.extract_detections(sequence, config, cache, None, None, 'cpu', versions)
                self.assertEqual(infer.call_count, 3)
                np.testing.assert_array_equal(first, second)
                self.assertEqual(len(first), 2)
                changed = dict(config, cache_score_threshold=0.1)
                with self.assertRaises(ValueError):
                    detector.load_cached_detections(sequence, changed, cache)
                frame = cache/sequence.name/'000002.npy'
                frame.write_bytes(b'corrupted')
                with self.assertRaises(ValueError):
                    detector.load_cached_detections(sequence, config, cache)
                detector.extract_detections(sequence, config, cache, None, None, 'cpu', versions)
                self.assertEqual(infer.call_count, 4)
                sequence.image_path(1).write_bytes(sequence.image_path(3).read_bytes())
                with self.assertRaises(ValueError):
                    detector.load_cached_detections(sequence, config, cache)


class NMSAdapterTests(unittest.TestCase):
    def test_peak_memory_available(self):
        self.assertGreater(detector.peak_process_memory_mb(), 0)

    def test_groups_and_restoration_on_exception(self):
        import torch
        import torchvision.ops.boxes as ops
        original = ops.batched_nms
        boxes = torch.tensor([[0.,0.,10.,10.], [0.,0.,10.,10.], [0.,0.,10.,10.]])
        scores = torch.tensor([.9,.8,.7])
        groups = torch.tensor([0,0,1])
        self.assertEqual(detector.own_batched_nms(boxes,scores,groups,.5).tolist(), [0,2])
        with self.assertRaises(RuntimeError):
            with detector.own_nms_context():
                self.assertIsNot(ops.batched_nms, original)
                raise RuntimeError('fixture')
        self.assertIs(ops.batched_nms, original)

    def test_actual_forward_never_calls_library_nms(self):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            # Sem pesos/download: testa o caminho real do detector, não sua acurácia.
            model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                        min_size=64, max_size=96, rpn_pre_nms_top_n_test=50,
                        rpn_post_nms_top_n_test=20).eval()
            with mock.patch.object(torch.ops.torchvision, 'nms', side_effect=AssertionError('NMS proibido')):
                with detector.own_nms_context() as counts, torch.inference_mode():
                    result = model([torch.rand(3,64,64)])
                self.assertGreaterEqual(counts['batched_nms'], 2)
                self.assertEqual(len(result), 1)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
