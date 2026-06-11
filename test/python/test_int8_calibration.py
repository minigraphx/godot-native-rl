import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import int8_calibration as cal  # noqa: E402

# Guarded heavy import (#141): missing numpy -> skips, not errors, under bare python.
try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class TestTableShapeArg(unittest.TestCase):
    def test_whc_order(self):
        self.assertEqual(cal.table_shape_arg(8, 8, 3), "[8,8,3]")
        self.assertEqual(cal.table_shape_arg(16, 4, 1), "[16,4,1]")


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestSampleImages(unittest.TestCase):
    def test_shape_dtype_range(self):
        imgs = cal.sample_images(5, 8, 8, 3, seed=0)
        self.assertEqual(imgs.shape, (5, 8, 8, 3))
        self.assertEqual(imgs.dtype.name, "uint8")

    def test_deterministic(self):
        import numpy as np
        a = cal.sample_images(4, 8, 8, 3, seed=7)
        b = cal.sample_images(4, 8, 8, 3, seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_seed_changes_data(self):
        import numpy as np
        a = cal.sample_images(4, 8, 8, 3, seed=1)
        b = cal.sample_images(4, 8, 8, 3, seed=2)
        self.assertFalse(np.array_equal(a, b))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestImageToChwFloat(unittest.TestCase):
    def test_chw_normalized(self):
        import numpy as np
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[..., 0] = 255  # red channel max
        chw = cal.image_to_chw_float(img)
        self.assertEqual(chw.shape, (3, 8, 8))
        self.assertEqual(chw.dtype.name, "float32")
        self.assertAlmostEqual(float(chw[0].max()), 1.0, places=5)
        self.assertAlmostEqual(float(chw[1].max()), 0.0, places=5)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestWriteCalibrationSet(unittest.TestCase):
    def test_writes_npy_and_list(self):
        import numpy as np
        imgs = cal.sample_images(3, 8, 8, 3, seed=0)
        with tempfile.TemporaryDirectory() as d:
            list_path = cal.write_calibration_set(imgs, d)
            self.assertTrue(list_path.is_file())
            lines = [ln for ln in list_path.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 3)
            arr = np.load(lines[0])
            self.assertEqual(arr.shape, (3, 8, 8))
            self.assertEqual(arr.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
