import os
import json
import tempfile
import logging
from fastapi import Request, HTTPException, status

logger = logging.getLogger("backend.auth_guard")

# ไฟล์ JSON สำหรับเก็บจำนวนการใช้งานของผู้ใช้ที่ไม่ได้ล็อกอิน (Guest) โดยใช้ IP Address
QUOTA_FILE = os.path.join(tempfile.gettempdir(), "harmoniq_guest_quota.json")


def _load_guest_quota() -> dict[str, int]:
    """อ่านข้อมูลโควตาของผู้ใช้ Guest จากไฟล์ JSON"""
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_guest_quota(data: dict[str, int]) -> None:
    """บันทึกข้อมูลโควตาของผู้ใช้ Guest ลงไฟล์ JSON"""
    try:
        with open(QUOTA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"ไม่สามารถบันทึกไฟล์โควตาได้: {e}")


def validate_tier_and_quota(user_tier: str, used_quota: int, model_type: str = "LSTM", pitch_shift_semitones: int = 0):
    """
    ตรวจสอบสิทธิ์การเข้าใช้งานและโควตาตามระดับสมาชิก
    รองรับการแยก Stem, Auto-EQ, Compressor และ Pitch Shifting
    เข้ากันได้กับ Python 3.10
    """
    tier_upper = (user_tier or "FREE").upper()
    model_upper = (model_type or "LSTM").upper()

    # 1. ตรวจสอบการล็อกโมเดล: CNN สงวนสิทธิ์เฉพาะผู้ใช้ที่อัปเกรดแล้วเท่านั้น
    if model_upper == "CNN" and tier_upper == "FREE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="โมเดล AutoEQ แบบ CNN สงวนสิทธิ์เฉพาะผู้ใช้สมาชิกระดับ Basic หรือ Pro เท่านั้น กรุณาอัปเกรดแพ็กเกจเพื่อปลดล็อก"
        )

    # 2. ตรวจสอบช่วง Pitch Shift ตามระดับสมาชิก
    max_pitch_shifts = {
        "FREE": 2,
        "BASIC": 6,
        "PRO": 12
    }
    max_allowed = max_pitch_shifts.get(tier_upper, 2)
    if abs(pitch_shift_semitones) > max_allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"การปรับ Pitch {pitch_shift_semitones} เซมิโทน เกินโควตาของแพ็กเกจ {tier_upper} (สูงสุด ±{max_allowed} เซมิโทน) กรุณาอัปเกรดแพ็กเกจเพื่อขยายขีดจำกัด"
        )

    # 3. ตรวจสอบโควตาการใช้งาน: Free=3, Basic=15, Pro=-1 (ไม่จำกัด)
    tier_limits = {
        "FREE": 3,
        "BASIC": 15,
        "PRO": -1
    }

    limit = tier_limits.get(tier_upper, 3)
    if limit != -1 and used_quota >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"โควตาประมวลผลฟรีสำหรับผู้ใช้ {tier_upper} เต็มแล้ว ({used_quota}/{limit} เพลง) กรุณาสมัครสมาชิกเพื่อใช้งานต่อ"
        )


def check_and_increment_quota(request: Request, user_tier: str, model_type: str = "LSTM", pitch_shift_semitones: int = 0):
    """
    ตรวจสอบสิทธิ์และนับจำนวนโควตาตาม IP Address สำหรับผู้ใช้ระดับ FREE (Guest)
    และบันทึกลงไฟล์ harmoniq_guest_quota.json ใน Temp directory
    """
    tier_upper = (user_tier or "FREE").upper()
    client_ip = request.client.host if request.client else "127.0.0.1"

    guest_data = _load_guest_quota()
    used_count = guest_data.get(client_ip, 0) if tier_upper == "FREE" else 0

    logger.info(f"[AUTH GUARD] IP: {client_ip} | Tier: {tier_upper} | Used: {used_count}/3")

    validate_tier_and_quota(user_tier=tier_upper, used_quota=used_count, model_type=model_type, pitch_shift_semitones=pitch_shift_semitones)

    if tier_upper == "FREE":
        guest_data[client_ip] = used_count + 1
        _save_guest_quota(guest_data)

