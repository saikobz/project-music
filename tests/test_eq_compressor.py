import os
import tempfile
import unittest

import torch
import torchaudio

from backend.eq_compressor import apply_compression


class TestEqCompressor(unittest.TestCase):
    def _write_input(self, path: str, sample_rate: int = 44100, seconds: float = 1.0) -> torch.Tensor:
        t = torch.arange(int(sample_rate * seconds), dtype=torch.float32) / sample_rate
        waveform = 0.9 * torch.sin(2 * torch.pi * 220.0 * t)
        waveform = waveform.unsqueeze(0)  # mono
        torchaudio.save(path, waveform, sample_rate)
        return waveform

    def test_apply_compression_accepts_custom_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            output_dir = os.path.join(temp_dir, "out")
            self._write_input(input_path)

            output_path = apply_compression(
                input_path,
                strength="medium",
                genre="pop",
                output_dir=output_dir,
                threshold=-30.0,
                ratio=6.0,
                attack=2.0,
                release=80.0,
                knee=8.0,
                makeup_gain=6.0,
                dry_wet=100.0,
                output_ceiling=-3.0,
            )

            self.assertTrue(os.path.exists(output_path))
            compressed, _ = torchaudio.load(output_path)
            peak = torch.max(torch.abs(compressed)).item()
            ceiling_lin = 10 ** (-3.0 / 20.0)
            self.assertLessEqual(peak, ceiling_lin + 1e-4)

    def test_apply_compression_dry_wet_zero_keeps_signal_close(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            output_dir = os.path.join(temp_dir, "out")
            original = self._write_input(input_path)

            output_path = apply_compression(
                input_path,
                strength="hard",
                genre="trap",
                output_dir=output_dir,
                threshold=-40.0,
                ratio=10.0,
                dry_wet=0.0,
            )

            processed, _ = torchaudio.load(output_path)
            diff = torch.mean(torch.abs(processed - original)).item()
            self.assertLess(diff, 1e-3)

    def test_invalid_strength_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            self._write_input(input_path)

            with self.assertRaises(ValueError):
                apply_compression(input_path, strength="extreme")


if __name__ == "__main__":
    unittest.main()
