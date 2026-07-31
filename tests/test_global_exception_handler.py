# regression tests สำหรับ C6: Global Exception Handler
# - Exception ที่ไม่คาดคิด -> 500 พร้อม shape {"status":"error","message":...}
# - ข้อความ generic (M17: ไม่รั่ว error detail ภายในออกไปที่ client)

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main
from backend.routers import audio_ops


class TestGlobalExceptionHandler(unittest.TestCase):
    def setUp(self) -> None:
        # raise_server_exceptions=False: FastAPI ส่ง Exception handler ไปให้ ServerErrorMiddleware
        # (TestClient default=True จะ re-raise exception ที่ ServerErrorMiddleware จับได้
        #  ทั้งที่ production คืน 500 JSON ปกติ)
        self.client = TestClient(main.app, raise_server_exceptions=False)
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

    def test_unhandled_exception_returns_generic_500(self) -> None:
        # RuntimeError (ไม่ใช่ HTTPException/ValueError/AutoEQModelLoadError)
        # ต้องถูกจับโดย global handler แทน try/except ใน route
        with patch.object(
            audio_ops, "save_upload", side_effect=RuntimeError("internal boom detail")
        ):
            response = self.client.post(
                "/apply-eq-ai?genre=pop",
                files={"file": ("song.wav", b"RIFF0000WAVE", "audio/wav")},
            )

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["status"], "error")
        # M17: ต้องไม่รั่ว error detail ภายใน (เดิมส่ง str(e) กลับไป)
        self.assertNotIn("internal boom detail", body["message"])
        self.assertNotIn("boom", body["message"])

    def test_http_exception_still_returns_its_status(self) -> None:
        # HTTPException (เช่น 400 จาก save_upload) ต้องไม่ถูก global handler กลืน
        with patch.object(
            audio_ops,
            "save_upload",
            side_effect=__import__("fastapi").HTTPException(status_code=400, detail="bad file"),
        ):
            response = self.client.post(
                "/apply-eq-ai?genre=pop",
                files={"file": ("song.wav", b"RIFF0000WAVE", "audio/wav")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "bad file")


if __name__ == "__main__":
    unittest.main()
