# regression tests สำหรับ B8: Trim validation ใน save_upload
# และ B9: convert_to_mp3 ต้อง raise แทนคืน wav path เงียบๆ

import asyncio
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torchaudio
from fastapi import HTTPException, UploadFile

from backend.services.storage import convert_to_mp3, save_upload


def _make_wav_bytes(sample_rate: int = 22050, seconds: float = 2.0) -> bytes:
    """สร้าง WAV ไฟล์จำลองความยาว seconds วินาที"""
    t = torch.arange(int(sample_rate * seconds), dtype=torch.float32) / sample_rate
    waveform = 0.5 * torch.sin(2 * torch.pi * 220.0 * t).unsqueeze(0)
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    return buffer.getvalue()


def _make_upload_file(data: bytes, filename: str = "song.wav") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), size=len(data), filename=filename)


class TestSaveUploadTrimValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_save(self, **kwargs):
        return asyncio.run(
            save_upload(_make_upload_file(_make_wav_bytes()), upload_dir=self.temp_dir.name, **kwargs)
        )

    def test_trim_start_negative_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run_save(trim_start=-1.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_trim_start_beyond_duration_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run_save(trim_start=5.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_trim_end_before_start_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run_save(trim_start=1.5, trim_end=0.5)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_trim_end_beyond_duration_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            self._run_save(trim_start=0.0, trim_end=10.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_valid_trim_truncates_file(self) -> None:
        _, input_path = self._run_save(trim_start=0.5, trim_end=1.5)
        self.assertTrue(os.path.exists(input_path))
        data, sr = torchaudio.load(input_path)
        duration = data.shape[-1] / sr
        self.assertAlmostEqual(duration, 1.0, delta=0.01)

    def test_no_trim_keeps_full_length(self) -> None:
        _, input_path = self._run_save()
        data, sr = torchaudio.load(input_path)
        self.assertAlmostEqual(data.shape[-1] / sr, 2.0, delta=0.01)

    def test_non_wav_extension_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(
                save_upload(_make_upload_file(b"not wav", filename="song.mp3"), upload_dir=self.temp_dir.name)
            )
        self.assertEqual(ctx.exception.status_code, 400)


class TestConvertToMp3(unittest.TestCase):
    def _make_segment(self) -> MagicMock:
        segment = MagicMock()

        def fake_export(mp3_path: str, *args, **kwargs) -> bool:
            with open(mp3_path, "wb") as file:
                file.write(b"mp3data")
            return True

        segment.export = fake_export
        return segment

    def test_success_removes_source_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "song.wav")
            with open(wav_path, "wb") as file:
                file.write(b"wavdata")

            with patch("pydub.AudioSegment.from_wav", return_value=self._make_segment()):
                out_path = convert_to_mp3(wav_path)

            self.assertEqual(out_path, os.path.join(temp_dir, "song.mp3"))
            self.assertTrue(os.path.exists(out_path))
            self.assertFalse(os.path.exists(wav_path))

    def test_success_keeps_source_when_remove_source_false(self) -> None:
        # regression B3: stem ต้นฉบับต้องไม่ถูกลบ
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "song.wav")
            with open(wav_path, "wb") as file:
                file.write(b"wavdata")

            with patch("pydub.AudioSegment.from_wav", return_value=self._make_segment()):
                out_path = convert_to_mp3(wav_path, remove_source=False)

            self.assertTrue(os.path.exists(out_path))
            self.assertTrue(os.path.exists(wav_path))

    def test_failure_raises_runtime_error_and_keeps_source(self) -> None:
        # regression B9: ไม่ควรคืน wav path แบบเงียบๆ เมื่อแปลงไม่สำเร็จ
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = os.path.join(temp_dir, "song.wav")
            with open(wav_path, "wb") as file:
                file.write(b"wavdata")

            with patch("pydub.AudioSegment.from_wav", side_effect=Exception("corrupt wav")):
                with self.assertRaises(RuntimeError):
                    convert_to_mp3(wav_path)

            self.assertTrue(os.path.exists(wav_path))


if __name__ == "__main__":
    unittest.main()
