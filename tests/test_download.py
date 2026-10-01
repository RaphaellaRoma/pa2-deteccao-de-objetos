from pathlib import Path
import tempfile
import unittest
import zipfile
from download_mot17 import extract_training


class DownloadTests(unittest.TestCase):
    def test_selective_extraction_no_traversal_or_duplicate_videos(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root)/'sample.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('MOT17/train/MOT17-09-FRCNN/seqinfo.ini', 'fixture')
                z.writestr('MOT17/train/MOT17-09-FRCNN/img1/000001.jpg', 'fixture')
                z.writestr('MOT17/train/MOT17-09-DPM/img1/000001.jpg', 'duplicate')
                z.writestr('MOT17/test/MOT17-01-FRCNN/seqinfo.ini', 'test')
                z.writestr('MOT17/train/MOT17-09-FRCNN/../../escape.txt', 'escape')
                z.writestr('MOT17/train/MOT17-02-FRCNN/seqinfo.ini', 'unselected')
            output = Path(root)/'data'
            self.assertEqual(extract_training(archive, output, ['MOT17-09']), 2)
            self.assertEqual(len(list(output.rglob('*.*'))), 2)
            self.assertFalse((Path(root)/'escape.txt').exists())
            self.assertEqual(extract_training(archive, output, ['MOT17-09']), 2)
