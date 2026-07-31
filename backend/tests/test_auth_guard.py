import json
import os
import tempfile
import unittest
from datetime import date
from unittest import mock
from fastapi import HTTPException
from backend.utils.auth_guard import (
    validate_tier_and_quota,
    validate_request_quota,
    increment_guest_quota,
    _load_guest_quota,
    _save_guest_quota,
)

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
        tmp = self.quota_file + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

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

    def test_guest_quota_full_is_blocked_at_validation(self):
        # Guest ที่ใช้ครบ 3 ในวันนี้ -> ถูกบล็อกด้วย 403 ในขั้น validate (ยังไม่นับเพิ่ม)
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": today}})
        with self.assertRaises(HTTPException) as cm:
            validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        self.assertEqual(cm.exception.status_code, 403)
        self.assertIn("โควตาประมวลผลฟรีสำหรับผู้ใช้ FREE เต็มแล้ว (3/3 เพลง)", cm.exception.detail)

    def test_guest_quota_resets_then_counts_again(self):
        # Guest ที่ใช้ครบ 3 เมื่อวาน -> วันนี้รีเซ็ตแล้วใช้ได้ และนับเพิ่มเป็น 1
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": "2020-01-01"}})
        validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        increment_guest_quota(request=self._make_request("203.0.113.5"))
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)
        self.assertEqual(data["203.0.113.5"]["date"], date.today().isoformat())

    def test_guest_increment_saved_to_file(self):
        # Guest ใช้ครั้งแรก -> นับเพิ่มเป็น 1 และบันทึกลงไฟล์
        validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        increment_guest_quota(request=self._make_request("203.0.113.5"))
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)

    def test_validate_does_not_increment(self):
        # B6: การ validate ต้องไม่หักโควตา (นับเมื่อ increment เท่านั้น)
        validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        # ไฟล์ที่ mkstemp สร้างไว้ต้องยังว่าง (validate ไม่เขียนข้อมูลใดๆ)
        self.assertEqual(os.path.getsize(self.quota_file), 0)

    def test_guest_header_tier_is_ignored(self):
        # B1: Guest ที่ส่ง X-User-Tier: PRO ต้องถูกคิดเป็น FREE เท่านั้น (ไม่ skip การนับ)
        validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="PRO")
        increment_guest_quota(request=self._make_request("203.0.113.5"))
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)

    def test_guest_header_pro_still_blocked_by_free_quota(self):
        # B1: Guest ที่ปลอม PRO ก็ยังโดนจำกัด 3 ครั้ง/วัน เหมือน FREE
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": today}})
        with self.assertRaises(HTTPException) as cm:
            validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="PRO")
        self.assertEqual(cm.exception.status_code, 403)

    def test_guest_pitch_shift_limited_to_free_range(self):
        # B1: Guest ที่ปลอม PRO ยังโดนจำกัด pitch ±2 เซมิโทนเหมือน FREE
        with self.assertRaises(HTTPException):
            validate_request_quota(
                request=self._make_request("203.0.113.5"),
                user_tier="PRO",
                pitch_shift_semitones=6,
            )

    def test_guest_cnn_model_locked(self):
        # B1: Guest ที่ปลอม BASIC ก็ยังใช้โมเดล CNN ไม่ได้
        with self.assertRaises(HTTPException):
            validate_request_quota(
                request=self._make_request("203.0.113.5"),
                user_tier="BASIC",
                model_type="CNN",
            )

    def test_save_quota_is_atomic_and_leaves_no_tmp(self):
        # B7: บันทึกผ่าน tmp + os.replace -> ไม่เหลือไฟล์ .tmp และ JSON ถูกต้อง
        validate_request_quota(request=self._make_request("203.0.113.5"), user_tier="FREE")
        increment_guest_quota(request=self._make_request("203.0.113.5"))
        self.assertFalse(os.path.exists(self.quota_file + ".tmp"))
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 1)

    def test_logged_in_user_skips_ip_tracking(self):
        # ผู้ใช้ที่ Login แล้ว (user_id) ต้องไม่ถูกบล็อกและไม่ถูกนับโควตาตาม IP
        today = date.today().isoformat()
        self._write_quota_file({"203.0.113.5": {"count": 3, "date": today}})
        validate_request_quota(
            request=self._make_request("203.0.113.5"), user_tier="FREE", user_id="user-123"
        )
        increment_guest_quota(request=self._make_request("203.0.113.5"), user_id="user-123")
        data = self._read_quota_file()
        self.assertEqual(data["203.0.113.5"]["count"], 3)

    def test_logged_in_user_still_blocked_for_cnn(self):
        # ผู้ใช้ที่ Login แล้วยังต้องโดนตรวจสอบสิทธิ์โมเดล (CNN ต้องไม่ใช้กับ FREE)
        with self.assertRaises(HTTPException) as cm:
            validate_request_quota(
                request=self._make_request("203.0.113.5"),
                user_tier="FREE",
                user_id="user-123",
                model_type="CNN",
            )
        self.assertEqual(cm.exception.status_code, 403)


class TestGuestQuotaAtomicSave(unittest.TestCase):
    """B7: การเขียนไฟล์โควตาแบบ atomic"""

    def setUp(self):
        fd, self.quota_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.patcher = mock.patch("backend.utils.auth_guard.QUOTA_FILE", self.quota_file)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.quota_file):
            os.remove(self.quota_file)
        tmp = self.quota_file + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)

    def test_save_replaces_file_with_valid_json(self):
        _save_guest_quota({"1.2.3.4": {"count": 1, "date": date.today().isoformat()}})
        with open(self.quota_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["1.2.3.4"]["count"], 1)
        self.assertFalse(os.path.exists(self.quota_file + ".tmp"))

    def test_corrupted_file_returns_empty_gracefully(self):
        # ไฟล์เสีย -> โหลดไม่ได้ -> คืน {} (โควตาเริ่มนับใหม่) แทนการ crash
        with open(self.quota_file, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        data = _load_guest_quota()
        self.assertEqual(data, {})


if __name__ == "__main__":
    unittest.main()
