# regression tests สำหรับ F12: /convert-format endpoint
# - แปลง wav -> mp3 (ไม่ต้องประมวลผลใหม่)
# - format เดียวกัน -> ส่งกลับไฟล์เดิม
# - ปฏิเสธนามสกุลอื่น

import io
import unittest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import backend.main as main
from backend.routers import audio_ops


class TestConvertFormatEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        # ปิดการเช็คและนับโควตา (endpoint นี้ไม่หักโควตาโดย design แต่กันผลกระทบข้าม test)
        self._quota_patch = patch.object(
            audio_ops, "validate_request_quota", new=lambda *args, **kwargs: None
        )
        self._increment_patch = patch.object(
            audio_ops, "increment_guest_quota", new=lambda *args, **kwargs: None
        )
        self._quota_patch.start()
        self._increment_patch.start()

    def tearDown(self) -> None:
        self._quota_patch.stop()
        self._increment_patch.stop()

    def test_convert_wav_to_mp3(self) -> None:
        fake_segment = MagicMock()

        def fake_export(output_path: str, *args, **kwargs) -> None:
            with open(output_path, "wb") as f:
                f.write(b"ID3-mp3data")

        fake_segment.export = fake_export

        with patch("pydub.AudioSegment.from_file", return_value=fake_segment):
            response = self.client.post(
                "/convert-format?export_format=mp3",
                files={"file": ("song.wav", b"RIFF0000WAVE", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/mpeg"))
        self.assertEqual(response.content, b"ID3-mp3data")
        self.assertIn("song.mp3", response.headers.get("content-disposition", ""))

    def test_same_format_returns_file_as_is(self) -> None:
        # wav -> wav: ไม่ต้องผ่าน pydub (ส่งไฟล์เดิมกลับ)
        with patch("pydub.AudioSegment.from_file") as mock_from_file:
            response = self.client.post(
                "/convert-format?export_format=wav",
                files={"file": ("song.wav", b"RIFF0000WAVE", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/wav"))
        self.assertEqual(response.content, b"RIFF0000WAVE")
        mock_from_file.assert_not_called()

    def test_rejects_non_audio_extension(self) -> None:
        response = self.client.post(
            "/convert-format?export_format=mp3",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_rejects_invalid_export_format(self) -> None:
        response = self.client.post(
            "/convert-format?export_format=flac",
            files={"file": ("song.wav", b"RIFF0000WAVE", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
