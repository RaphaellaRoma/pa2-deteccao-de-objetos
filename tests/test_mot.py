import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from mot_data import (Sequence, load_config, valid_gt, evaluation_mask,
                      save_predictions, load_predictions)
from detection_metrics import evaluate_detection
from parte1 import evaluate_sequence, evaluate_all, run_tracker


def fixture(root, name='MOT17-09', length=3):
    d = Path(root)/'train'/f'{name}-FRCNN'
    (d/'gt').mkdir(parents=True)
    (d/'det').mkdir()
    (d/'img1').mkdir()
    (d/'seqinfo.ini').write_text(f'[Sequence]\nname={name}-FRCNN\nimDir=img1\nframeRate=30\nseqLength={length}\nimWidth=64\nimHeight=64\nimExt=.jpg\n')
    (d/'gt/gt.txt').write_text(''.join(f'{f},1,11,11,10,10,1,1,0\n' for f in range(1, length+1)))
    (d/'det/det.txt').write_text(''.join(f'{f},-1,11,11,10,10,.9,-1,-1,-1\n' for f in range(1, length+1)))
    for f in range(1, length+1):
        Image.new('RGB', (64, 64), (f*20, 0, 0)).save(d/'img1'/f'{f:06d}.jpg')
    return Sequence.load(root, name)


class MOTTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_coordinate_roundtrip_and_zero_visibility(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            gt = s.ground_truth()
            self.assertEqual(gt[0, 2], 10)
            self.assertEqual(len(valid_gt(gt)), 3)
            pred = gt[:, :6]
            save_predictions(Path(root)/'pred.txt', pred)
            np.testing.assert_array_equal(load_predictions(Path(root)/'pred.txt'), pred)

    def test_ignore_priority(self):
        raw = np.array([[1,1,0,0,10,10,1,1,0],
                        [1,2,0,0,10,10,1,8,1],
                        [1,3,30,0,10,10,0,1,1],
                        [1,4,60,0,10,10,1,3,1]])
        boxes = [[0,0,10,10], [0,0,10,10], [30,0,10,10], [60,0,10,10]]
        np.testing.assert_array_equal(evaluation_mask(raw, [1]*4, boxes), [True,False,False,True])

    def test_perfect_pipeline(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            result, pred = evaluate_sequence(s, s.public_detections(), self.config)
            self.assertEqual(result['idf1'], 1)
            self.assertEqual(result['map_50_95'], 1)
            self.assertEqual(len(pred), 3)
            self.assertEqual(result['identity_ratio'], 1)

    def test_gt_never_filters_tracker_input(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            # Pessoa parada: inferência cria ID, avaliação o ignora.
            with (s.directory/'gt/gt.txt').open('a') as stream:
                stream.write('1,2,31,11,10,10,1,7,1\n')
            det = np.vstack([s.public_detections(), [1,30,10,10,10,.9]])
            result, pred = evaluate_sequence(s, det, self.config)
            self.assertEqual(result['raw_predicted_identities'], 2)
            self.assertEqual(result['predicted_identities'], 1)
            self.assertEqual(len(pred), 4)

    def test_empty_frames_and_partial_flag(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            det = s.public_detections()[[0,2]]
            result, pred = evaluate_sequence(s, det, self.config)
            self.assertEqual(result['predicted_identities'], 1)
            self.assertEqual(result['fragmentations'], 1)
            partial, _ = evaluate_sequence(s, det, self.config, max_frames=1)
            self.assertFalse(partial['full_sequence'])

    def test_overlap_determinism(self):
        a = np.array([[1,0,0,10,10,.9], [1,1,0,10,10,.9],
                      [2,0,0,10,10,.9], [2,1,0,10,10,.9]])
        np.testing.assert_array_equal(run_tracker(a,2,self.config), run_tracker(a,2,self.config))

    def test_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            c = dict(self.config)
            c['split'] = {'train':['MOT17-09'], 'validation':['MOT17-09']}
            p = Path(root)/'config.json'
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                load_config(p)

    def test_cli_equivalent_and_completeness(self):
        with tempfile.TemporaryDirectory() as root:
            s = fixture(root)
            config = dict(self.config)
            config['split'] = {'train':['MOT17-05'], 'validation':['MOT17-09']}
            direct, _ = evaluate_sequence(s, s.public_detections(), config)
            results = evaluate_all(root, config, Path(root)/'output', split='validation')
            self.assertEqual(results[0], direct)
            payload = json.loads((Path(root)/'output/results.json').read_text())
            self.assertFalse(payload['complete_part1'])
            self.assertTrue((Path(root)/'output/descolamento.png').exists())


class APTests(unittest.TestCase):
    def setUp(self):
        self.gt = [[1,1,0,0,10,10]]

    def test_perfect_and_missing(self):
        self.assertEqual(evaluate_detection(self.gt, [[1,0,0,10,10,.9]])['map_50_95'], 1)
        self.assertEqual(evaluate_detection(self.gt, [])['ap50'], 0)
        self.assertIsNone(evaluate_detection([], [])['ap50'])

    def test_high_score_fp_penalty(self):
        det = [[1,40,0,10,10,.99], [1,0,0,10,10,.9]]
        self.assertEqual(evaluate_detection(self.gt, det)['ap50'], 0.5)

    def test_no_cross_frame_matching(self):
        self.assertEqual(evaluate_detection(self.gt, [[2,0,0,10,10,.9]])['ap50'], 0)

    def test_duplicate_cannot_recover_two_gt(self):
        gt = [self.gt[0], [1,2,40,0,10,10]]
        det = [[1,0,0,10,10,.9], [1,0,0,10,10,.8]]
        self.assertLess(evaluate_detection(gt, det)['ap50'], 0.51)


if __name__ == '__main__':
    unittest.main()
