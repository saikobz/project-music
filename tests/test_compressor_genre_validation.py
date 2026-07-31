# regression tests สำหรับ B2: Path Traversal ผ่าน genre/model_id param
# - endpoint ต้อง reject genre/model_id ที่ไม่ตรง whitelist ด้วย 422 ก่อนถึง business logic
# - apply_compression ต้อง reject genre ที่ไม่รองรับด้วย ValueError (defense-in-depth)

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.eq_compressor import apply_compression
from backend.routers import audio_ops
from backend.services.storage import UPLOAD_DIR


class TestCompressorGenreValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        # ปิดการเช็คและนับโควตา เพื่อให้เทสไม่ชน quota ตาม IP
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

    def test_compressor_endpoint_rejects_path_traversal_genre(self) -> None:
        # ส่ง genre ที่มี path separator -> ต้องโดน 422 จาก pattern validation ก่อนเขียนไฟล์
        for evil_genre in ("../../evil", "..\\evil", "a/b", "genre;rm"):
            response = self.client.post(
                f"/apply-compressor?genre={evil_genre}",
                files={"file": ("song.wav", b"abc", "audio/wav")},
            )
            self.assertEqual(response.status_code, 422, f"genre={evil_genre}")

    def test_compressor_endpoint_rejects_unknown_genre(self) -> None:
        response = self.client.post(
            "/apply-compressor?genre=not-a-genre",
            files={"file": ("song.wav", b"abc", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)

    def test_compressor_endpoint_accepts_valid_genres(self) -> None:
        # ตรวจว่า whitelist เดิมยังใช้ได้ครบทุกตัว (ไม่ทำให้ regex หลุด)
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.wav")
            output_path = os.path.join(temp_dir, "output.wav")

            async def fake_save_upload(file, upload_dir=UPLOAD_DIR, trim_start=None, trim_end=None):
                return "test-id", input_path

            def fake_apply_compression(*args, **kwargs) -> str:
                with open(output_path, "wb") as file:
                    file.write(b"RIFF0000WAVEfmt ")
                return output_path

            for genre in ("general", "pop", "rock", "trap", "country", "soul"):
                with open(input_path, "wb") as file:
                    file.write(b"dummy")
                with open(output_path, "wb") as file:
                    file.write(b"RIFF0000WAVEfmt ")
                with patch.object(audio_ops, "save_upload", new=fake_save_upload), patch.object(
                    audio_ops, "apply_compression", side_effect=fake_apply_compression
                ):
                    response = self.client.post(
                        f"/apply-compressor?strength=medium&genre={genre}",
                        files={"file": ("song.wav", b"abc", "audio/wav")},
                    )
                self.assertEqual(response.status_code, 200, f"genre={genre}")


class TestAutoEqGenreModelValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        # ปิดการเช็คและนับโควตา เพื่อให้เทสไม่ชน quota ตาม IP
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

    def test_eq_ai_endpoint_rejects_path_traversal_genre(self) -> None:
        response = self.client.post(
            "/apply-eq-ai?genre=../../evil",
            files={"file": ("song.wav", b"abc", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)

    def test_eq_ai_endpoint_rejects_unknown_model_id(self) -> None:
        response = self.client.post(
            "/apply-eq-ai?model_id=../../evil",
            files={"file": ("song.wav", b"abc", "audio/wav")},
        )
        self.assertEqual(response.status_code, 422)


class TestApplyCompressionGenreUnit(unittest.TestCase):
    def test_invalid_genre_raises_value_error_before_any_io(self) -> None:
        # ตรวจว่า genre ที่ไม่รองรับถูก reject ก่อนโหลดไฟล์/สร้างชื่อไฟล์
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_input = os.path.join(temp_dir, "does_not_exist.wav")
            with self.assertRaises(ValueError):
                apply_compression(
                    missing_input, strength="medium", genre="../../evil", output_dir=temp_dir
                )


if __name__ == "__main__":
    unittest.main()
