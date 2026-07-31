import json
import os
import tempfile
import unittest
from datetime import date
from unittest import mock
from fastapi import HTTPException
from backend.utils.auth_guard import validate_tier_and_quota, check_and_increment_quota, _load_guest_quota

class TestAuthGuard(unittest.TestCase):
    def test_free_tier_cannot_use_cnn_model(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=0, model_type="CNN")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("โมเดล AutoEQ แบบ CNN สงวนสิทธิ์เฉพาะผู้ใช้สมาชิกระดับ Basic หรือ Pro", cm.exception.detail)

    def test_free_tier_exceeded_quota(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=3, model_type="LSTM")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("โควตาประมวลผลฟรีสำหรับผู้ใช้ FREE เต็มแล้ว", cm.exception.detail)

    def test_free_tier_within_quota(self):
        # Should not raise exception when used_quota is 2 (< 3)
        validate_tier_and_quota(user_tier="FREE", used_quota=2, model_type="LSTM")

    def test_pitch_shift_range_limit_free_tier(self):
        with self.assertRaises(HTTPException) as cm:
            validate_tier_and_quota(user_tier="FREE", used_quota=0, model_type="LSTM", pitch_shift_semitones=5)
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("การปรับ Pitch 5 เซมิโทน เกินโควตาของแพ็กเกจ FREE", cm.exception.detail)

    def test_basic_tier_can_use_cnn_and_higher_pitch_shift(self):
        # Should not raise exception
        validate_tier_and_quota(user_tier="BASIC", used_quota=0, model_type="CNN", pitch_shift_semitones=5)

    def test_pro_tier_unlimited_quota(self):
        # Should not raise exception even with high used_quota and full octave pitch shift
        validate_tier_and_quota(user_tier="PRO", used_quota=999, model_type="CNN", pitch_shift_semitones=12)


class TestGuestQuota(unittest.TestCase):
    """ทดสอบโควตาของผู้ใช้ Guest: การรีเซ็ตรายวัน, การย้าย format เก่า และการแยกผู้ใช้ที่ Login แล้ว"""

    def setUp(self):
        # ใช้ไฟล์ JSON ชั่วคราวแทน QUOTA_FILE จริง เพื่อไม่ให้กระทบข้อมูลในเครื่อง
        fd, self.quota_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.patcher = mock.patch("backend.utils.auth_guard.QUOTA_FILE", self.quota_file)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.quota_file):
            os.remove(self.quota_file)

    def _write_quota_file(self, data: dict) -> None:
        with open(self.quota_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _read_quota_file(self) -> dict:
        with open(self.quota_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _make_request(self, ip: str):
        request = mock.MagicMock()
        request.client.host = ip
        return request

    def test_guest_old_date_is_reset_to_zero(self):
        # โควตาวันก่อนหน้า -> โหลดแล้วต้องรีเซ็ตเป็น 0 (รีเซ็ตรายวัน)
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": "2020-01-01"}})
        data = _load_guest_quota()
        self.assertEqual(data["203.0.113.5"]["count"], 0)
        self.assertEqual(data["203.0.113.5"]["date"], date.today().isoformat())

    def test_guest_same_day_is_not_reset(self):
        # โควตาวันเดียวกัน -> ต้องไม่ถูกรีเซ็ต
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 2, "date": today}})
        data = _load_guest_quota()
        self.assertEqual(data["203.0.113.5"]["count"], 2)

    def test_guest_old_format_migration(self):
        # format เก่า {"ip": count} -> ต้องย้ายเป็น format ใหม่ได้โดยยังนับค่าเดิม
        self._write_quota_file({"203.0.113.5": 3})
        data = _load_guest_quota()
        self.assertEqual(data["203.0.113.5"]["count"], 3)
        self.assertEqual(data["203.0.113.5"]["date"], date.today().isoformat())

    def test_guest_quota_full_is_blocked(self):
        # Guest ที่ใช้ครบ 3 ในวันนี้ -> ถูกบล็อกด้วย 403
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": today}})
        with self.assertRaises(HTTPException) as cm:
            check_and_increment_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("โควตาประมวลผลฟรีสำหรับผู้ใช้ FREE เต็มแล้ว (3/3 เพลง)", cm.exception.detail)

    def test_guest_quota_resets_then_counts_again(self):
        # Guest ที่ใช้ครบ 3 เมื่อวาน -> วันนี้รีเซ็ตแล้วใช้ได้ และนับเพิ่มเป็น 1
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": "2020-01-01"}})
        check_and_increment_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)
        self.assertEqual(data["203.0.113.5"]["date"], date.today().isoformat())

    def test_guest_increment_saved_to_file(self):
        # Guest ใช้ครั้งแรก -> นับเพิ่มเป็น 1 และบันทึกลงไฟล์
        check_and_increment_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)

    def test_logged_in_user_skips_ip_tracking(self):
        # ผู้ใช้ที่ Login แล้ว (มี user_id) ต้องไม่ถูกบล็อกและไม่ถูกนับโควตาตาม IP
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": today}})
        check_and_increment_quota(request=self._make_request("203.0.113.5"), user_tier="FREE", user_id="user-123")
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 3)

    def test_logged_in_user_still_blocked_for_cnn(self):
        # ผู้ใช้ที่ Login แล้วยังต้องโดนตรวจสอบสิทธิ์โมเดล (CNN ต้องไม่ใช้กับ FREE)
        with self.assertRaises(HTTPException) as cm:
            check_and_increment_quota(request=self._make_request("203.0.113.5"), user_tier="FREE", user_id="user-123", model_type="CNN")
        self.assertEqual(cm.exception.status_code, 403)

if __name__ == "__main__":
    unittest.main()
