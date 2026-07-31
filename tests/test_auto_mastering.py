# regression tests สำหรับ TU-3 (DSP Correctness):
# - B5: ไฟล์เงียบสนิท -> ต้องคงความเงียบไว้ ไม่ใช่ NaN/garbage full-scale DC (เดิมได้ 1.0 ทั้งไฟล์)
# - B4 guard: stereo processing ต้องเปลี่ยนสัญญาณจริง (ยืนยัน pedalboard 0.9.8 ทำงานถูกกับ (samples, channels))

import os
import tempfile
import unittest

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from backend.auto_mastering import apply_lufs_mastering, polish_vocal_file

SR = 44100


def _write_wav(path: str, data: np.ndarray) -> None:
    sf.write(path, data, SR)


class TestApplyLufsMasteringSilent(unittest.TestCase):
    """B5: silence ต้องไม่กลายเป็น NaN หรือสัญญาณเต็มสเกล"""

    def _assert_silent_output(self, data: np.ndarray) -> None:
        self.assertFalse(np.isnan(data).any(), "output ต้องไม่มี NaN")
        self.assertFalse(np.isinf(data).any(), "output ต้องไม่มี inf")
        max_val = float(np.max(np.abs(data)))
        self.assertLess(max_val, 1e-7, f"ไฟล์เงียบต้องเงียบอยู่ (ได้ max={max_val})")

    def test_silent_stereo_stays_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "silent.wav")
            output_path = os.path.join(temp_dir, "out.wav")
            _write_wav(input_path, np.zeros((SR * 2, 2), dtype=np.float32))

            apply_lufs_mastering(input_path, output_path, -14.0)

            data, sr = sf.read(output_path)
            self.assertEqual(sr, SR)
            self.assertEqual(data.shape, (SR * 2, 2))
            self._assert_silent_output(data)

    def test_silent_mono_stays_silent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "silent.wav")
            output_path = os.path.join(temp_dir, "out.wav")
            _write_wav(input_path, np.zeros(SR * 2, dtype=np.float32))

            apply_lufs_mastering(input_path, output_path, -14.0)

            data, _ = sf.read(output_path)
            self._assert_silent_output(data)

    def test_quiet_tone_does_not_produce_full_scale_output(self) -> None:
        # สัญญาณเบามาก (แต่ไม่เงียบสนิท) ต้องไม่กลายเป็น DC เต็มสเกลแบบ bug เดิม
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "quiet.wav")
            output_path = os.path.join(temp_dir, "out.wav")
            t = np.arange(SR * 2) / SR
            quiet = (0.001 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            _write_wav(input_path, np.stack([quiet, quiet], axis=1))

            apply_lufs_mastering(input_path, output_path, -14.0)

            data, _ = sf.read(output_path)
            self.assertFalse(np.isnan(data).any())
            self.assertLess(float(np.max(np.abs(data))), 1.0)


class TestApplyLufsMasteringNormalSignal(unittest.TestCase):
    def test_normal_signal_output_finite_and_moves_toward_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "tone.wav")
            output_path = os.path.join(temp_dir, "out.wav")
            t = np.arange(SR * 8) / SR
            tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            _write_wav(input_path, np.stack([tone, tone], axis=1))

            apply_lufs_mastering(input_path, output_path, -14.0)

            data, _ = sf.read(output_path)
            self.assertFalse(np.isnan(data).any())
            self.assertFalse(np.isinf(data).any())
            self.assertLessEqual(float(np.max(np.abs(data))), 1.0)

            output_lufs = pyln.Meter(SR).integrated_loudness(data)
            # window สั้นอาจวัดคลาดเคลื่อนได้บ้าง แต่ต้องเข้าใกล้ target มากพอ
            self.assertLess(abs(float(output_lufs) - (-14.0)), 6.0)


class TestPolishVocalFile(unittest.TestCase):
    """B4 guard: ยืนยัน chain ทำงานจริงกับ stereo (ไม่ใช่ no-op)"""

    def _make_stereo(self, amplitude: float) -> np.ndarray:
        t = np.arange(SR * 2) / SR
        low = amplitude * np.sin(2 * np.pi * 440 * t)
        high = amplitude * np.sin(2 * np.pi * 15000 * t)
        return np.stack([low, high], axis=1).astype(np.float32)

    def test_stereo_polish_changes_signal_and_preserves_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "vocal.wav")
            output_path = os.path.join(temp_dir, "polished.wav")
            original = self._make_stereo(0.9)
            _write_wav(input_path, original)

            polish_vocal_file(input_path, output_path)

            processed, sr = sf.read(output_path)
            self.assertEqual(sr, SR)
            self.assertEqual(processed.shape, original.shape)
            self.assertFalse(np.isnan(processed).any())
            self.assertFalse(np.isinf(processed).any())
            # HighShelfFilter +2dB ที่ 15kHz ต้องทำให้สัญญาณเปลี่ยนอย่างเห็นได้ชัด
            diff_rms = float(np.sqrt(np.mean((processed - original) ** 2)))
            self.assertGreater(diff_rms, 1e-3)

    def test_mono_polish_works(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "vocal.wav")
            output_path = os.path.join(temp_dir, "polished.wav")
            t = np.arange(SR * 2) / SR
            mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            _write_wav(input_path, mono)

            polish_vocal_file(input_path, output_path)

            processed, _ = sf.read(output_path)
            self.assertEqual(processed.shape, mono.shape)
            self.assertFalse(np.isnan(processed).any())
            self.assertFalse(np.isinf(processed).any())


if __name__ == "__main__":
    unittest.main()
