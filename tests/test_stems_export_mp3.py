# regression tests สำหรับ B3: Export stems แบบ MP3 ต้องไม่ลบ WAV ต้นฉบับ
# (เดิม convert_to_mp3 ลบ wav ทุกครั้ง ทำให้ player/karaoke/vocal-polish พังก่อน TTL)

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.routers import stems
from backend.services.job_manager import job_manager


def _fake_convert_to_mp3(wav_path: str, remove_source: bool = True) -> str:
    """จำลองการแปลง MP3: สร้างไฟล์ .mp3 ข้าง ๆ โดยไม่แตะ wav ต้นฉบับ"""
    mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
    with open(mp3_path, "wb") as file:
        file.write(b"mp3data")
    return mp3_path


class TestStemsExportKeepsOriginalWav(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_id = "test-export-mp3-keep"
        job_manager.register_job(self.file_id, self.temp_dir.name)
        # ปิดการเช็คและนับโควตา เพื่อให้เทสไม่ชน quota ตาม IP
        self._quota_patch = patch.object(
            stems, "validate_request_quota", new=lambda *args, **kwargs: None
        )
        self._increment_patch = patch.object(
            stems, "increment_guest_quota", new=lambda *args, **kwargs: None
        )
        self._quota_patch.start()
        self._increment_patch.start()

    def tearDown(self) -> None:
        self._quota_patch.stop()
        self._increment_patch.stop()
        job_manager._jobs.pop(self.file_id, None)
        self.temp_dir.cleanup()

    def _write_stem(self, name: str) -> None:
        with open(os.path.join(self.temp_dir.name, name), "wb") as file:
            file.write(b"RIFF0000WAVEfmt ")

    def test_export_stems_mp3_keeps_original_wav_files(self) -> None:
        # regression B3: หลัง export MP3 แล้ว wav ต้นฉบับต้องยังอยู่
        self._write_stem("vocals.wav")
        self._write_stem("drums.wav")

        with patch.object(stems, "convert_to_mp3", side_effect=_fake_convert_to_mp3):
            response = self.client.post(
                "/api/process/export",
                params={
                    "file_id": self.file_id,
                    "export_type": "stems",
                    "export_format": "mp3",
                    "stems": ["vocals", "drums"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], "zip")
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "vocals.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "drums.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "vocals.mp3")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "drums.mp3")))

    def test_export_single_stem_mp3_keeps_original_wav(self) -> None:
        self._write_stem("vocals.wav")

        with patch.object(stems, "convert_to_mp3", side_effect=_fake_convert_to_mp3):
            response = self.client.post(
                "/api/process/export",
                params={
                    "file_id": self.file_id,
                    "export_type": "stems",
                    "export_format": "mp3",
                    "stems": ["vocals"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["type"], "file")
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "vocals.wav")))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir.name, "vocals.mp3")))

    def test_export_stems_mp3_passes_remove_source_false(self) -> None:
        # ตรวจว่า caller ส่ง remove_source=False ให้ convert_to_mp3 จริง ๆ
        self._write_stem("vocals.wav")
        captured: list[tuple] = []

        def spy_convert(wav_path: str, remove_source: bool = True) -> str:
            captured.append((wav_path, remove_source))
            return _fake_convert_to_mp3(wav_path, remove_source)

        with patch.object(stems, "convert_to_mp3", side_effect=spy_convert):
            self.client.post(
                "/api/process/export",
                params={
                    "file_id": self.file_id,
                    "export_type": "stems",
                    "export_format": "mp3",
                    "stems": ["vocals"],
                },
            )

        self.assertEqual(len(captured), 1)
        self.assertFalse(captured[0][1])

    def test_export_unknown_file_id_returns_404(self) -> None:
        response = self.client.post(
            "/api/process/export",
            params={
                "file_id": "does-not-exist",
                "export_type": "stems",
                "export_format": "wav",
                "stems": ["vocals"],
            },
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
