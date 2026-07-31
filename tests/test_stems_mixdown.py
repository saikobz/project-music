# regression tests สำหรับ C2: mixdown_stems helper
# - รวม mono หลายไฟล์, จัดการ mono+stereo ปนกัน, normalize peak, ไฟล์ว่าง -> ValueError

import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from backend.routers.stems import mixdown_stems, create_zip_archive


def _write_tone(path: str, amplitude: float, stereo: bool = False) -> None:
    sr = 8000
    t = np.arange(sr // 2) / sr
    mono = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if stereo:
        data = np.stack([mono, mono * 0.5], axis=1)
    else:
        data = mono
    sf.write(path, data, sr)


class TestMixdownStems(unittest.TestCase):
    def test_mixes_mono_files_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = os.path.join(temp_dir, "a.wav")
            p2 = os.path.join(temp_dir, "b.wav")
            _write_tone(p1, 0.9)
            _write_tone(p2, 0.9)

            mix, sr = mixdown_stems([p1, p2])

            self.assertEqual(sr, 8000)
            self.assertEqual(mix.ndim, 1)
            # 0.9 + 0.9 = 1.8 -> ต้องถูก normalize ไม่เกิน 1.0
            self.assertLessEqual(float(np.max(np.abs(mix))), 1.0 + 1e-6)

    def test_mono_and_stereo_mix_becomes_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mono = os.path.join(temp_dir, "mono.wav")
            stereo = os.path.join(temp_dir, "stereo.wav")
            _write_tone(mono, 0.5)
            _write_tone(stereo, 0.5, stereo=True)

            mix, _ = mixdown_stems([mono, stereo])

            self.assertEqual(mix.ndim, 2)
            self.assertEqual(mix.shape[1], 2)

    def test_empty_list_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            mixdown_stems([])


class TestCreateZipArchive(unittest.TestCase):
    def test_creates_zip_with_arcnames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "song.wav")
            with open(src, "wb") as f:
                f.write(b"wavdata")
            zip_path = os.path.join(temp_dir, "out.zip")

            create_zip_archive(zip_path, [(src, "nested/song.wav")])

            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                self.assertIn("nested/song.wav", zf.namelist())
                self.assertEqual(zf.read("nested/song.wav"), b"wavdata")


if __name__ == "__main__":
    unittest.main()
