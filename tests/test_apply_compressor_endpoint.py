import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main


class TestApplyCompressorEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_apply_compressor_passes_custom_params(self) -> None:
        input_path = os.path.join(self.temp_dir.name, "input.wav")
        output_path = os.path.join(self.temp_dir.name, "output.wav")
        with open(input_path, "wb") as file:
            file.write(b"dummy")

        captured: dict[str, object] = {}

        async def fake_save_upload(file, upload_dir=main.UPLOAD_DIR):
            return "test-id", input_path

        def fake_apply_compression(input_file: str, strength: str, genre: str, output_dir: str, **kwargs) -> str:
            captured["input_file"] = input_file
            captured["strength"] = strength
            captured["genre"] = genre
            captured["output_dir"] = output_dir
            captured.update(kwargs)
            with open(output_path, "wb") as file:
                file.write(b"RIFF0000WAVEfmt ")
            return output_path

        with patch.object(main, "save_upload", new=fake_save_upload), patch.object(
            main, "apply_compression", side_effect=fake_apply_compression
        ):
            response = self.client.post(
                "/apply-compressor?strength=hard&genre=rock&threshold=-18&ratio=3&attack=4&release=120&knee=7&makeup_gain=2&dry_wet=75&output_ceiling=-1",
                files={"file": ("song.wav", b"abc", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers.get("content-type", "").startswith("audio/wav"))
        self.assertEqual(response.content, b"RIFF0000WAVEfmt ")
        self.assertEqual(captured["input_file"], input_path)
        self.assertEqual(captured["strength"], "hard")
        self.assertEqual(captured["genre"], "rock")
        self.assertEqual(captured["output_dir"], "compressed")
        self.assertEqual(captured["threshold"], -18.0)
        self.assertEqual(captured["ratio"], 3.0)
        self.assertEqual(captured["attack"], 4.0)
        self.assertEqual(captured["release"], 120.0)
        self.assertEqual(captured["knee"], 7.0)
        self.assertEqual(captured["makeup_gain"], 2.0)
        self.assertEqual(captured["dry_wet"], 75.0)
        self.assertEqual(captured["output_ceiling"], -1.0)
        self.assertFalse(os.path.exists(input_path))

    def test_apply_compressor_returns_400_on_value_error(self) -> None:
        input_path = os.path.join(self.temp_dir.name, "input.wav")
        with open(input_path, "wb") as file:
            file.write(b"dummy")

        async def fake_save_upload(file, upload_dir=main.UPLOAD_DIR):
            return "test-id", input_path

        def fake_apply_compression(*args, **kwargs) -> str:
            raise ValueError("invalid params")

        with patch.object(main, "save_upload", new=fake_save_upload), patch.object(
            main, "apply_compression", side_effect=fake_apply_compression
        ):
            response = self.client.post(
                "/apply-compressor?strength=medium&genre=pop",
                files={"file": ("song.wav", b"abc", "audio/wav")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "invalid params")
        self.assertFalse(os.path.exists(input_path))


if __name__ == "__main__":
    unittest.main()
