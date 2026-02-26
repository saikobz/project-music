import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.auto_eq_inference import AutoEQModelLoadError


class TestApplyEqAiEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_apply_eq_ai_returns_503_when_model_unavailable(self) -> None:
        input_path = os.path.join(self.temp_dir.name, "input.wav")
        with open(input_path, "wb") as file:
            file.write(b"dummy")

        async def fake_save_upload(file, upload_dir=main.UPLOAD_DIR):
            return "test-id", input_path

        def fake_apply_auto_eq_file(input_file: str, output_file: str) -> str:
            raise AutoEQModelLoadError("model mismatch")

        with patch.object(main, "save_upload", new=fake_save_upload), patch.object(
            main, "apply_auto_eq_file", side_effect=fake_apply_auto_eq_file
        ):
            response = self.client.post(
                "/apply-eq-ai?genre=trap",
                files={"file": ("song.wav", b"abc", "audio/wav")},
            )

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["status"], "error")
        self.assertEqual(body["error_code"], "AUTO_EQ_MODEL_UNAVAILABLE")
        self.assertIn("model mismatch", body["message"])
        self.assertFalse(os.path.exists(input_path))

    def test_apply_eq_ai_returns_audio_on_success(self) -> None:
        input_path = os.path.join(self.temp_dir.name, "input.wav")
        output_path = os.path.join(self.temp_dir.name, "output.wav")
        with open(input_path, "wb") as file:
            file.write(b"dummy")

        async def fake_save_upload(file, upload_dir=main.UPLOAD_DIR):
            return "test-id", input_path

        def fake_apply_auto_eq_file(input_file: str, requested_output_path: str) -> str:
            with open(output_path, "wb") as file:
                file.write(b"RIFF0000WAVEfmt ")
            return output_path

        with patch.object(main, "save_upload", new=fake_save_upload), patch.object(
            main, "apply_auto_eq_file", side_effect=fake_apply_auto_eq_file
        ):
            response = self.client.post(
                "/apply-eq-ai?genre=trap",
                files={"file": ("song.wav", b"abc", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/wav"))
        self.assertEqual(response.content, b"RIFF0000WAVEfmt ")
        self.assertFalse(os.path.exists(input_path))


if __name__ == "__main__":
    unittest.main()
