import os
import json
import tempfile
import logging
from datetime import date
from fastapi import Request, HTTPException, status

logger = logging.getLogger("backend.auth_guard")

# ไฟล์ JSON สำหรับเก็บจำนวนการใช้งานของผู้ใช้ที่ไม่ได้ล็อกอิน (Guest) โดยใช้ IP Address
QUOTA_FILE = os.path.join(tempfile.gettempdir(), "harmoniq_guest_quota.json")


def _load_guest_quota() -> dict[str, dict]:
    """
    อ่านข้อมูลโควตาของผู้ใช้ Guest จากไฟล์ JSON
    พร้อมรีเซ็ตโควตาที่หมดอายุรายวัน (อิงวันที่ในระบบ) และรองรับการย้าย format เก่า {"ip": count}
    """
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            today = date.today().isoformat()
            data = {}
            for ip, value in raw.items():
                if isinstance(value, int):
                    # รองรับ format เก่า {"ip": count} -> แปลงเป็น format ใหม่ {"ip": {"count": n, "date": "..."}}
                    data[ip] = {"count": value, "date": today}
                elif isinstance(value, dict):
                    entry = dict(value)
                    if entry.get("date") != today:
                        # โควตาหมดอายุรายวัน -> รีเซ็ตเป็น 0
                        entry = {"count": 0, "date": today}
                    data[ip] = entry
            return data
        except Exception:
            pass
    return {}


def _save_guest_quota(data: dict[str, dict]) -> None:
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


def check_and_increment_quota(request: Request, user_tier: str, user_id: str | None = None, model_type: str = "LSTM", pitch_shift_semitones: int = 0):
    """
    ตรวจสอบสิทธิ์และนับจำนวนโควตา
    - ผู้ใช้ที่ Login แล้ว (มี user_id): ข้ามการนับโควตาตาม IP (โควตาจัดการผ่าน Database ฝั่ง Frontend)
    - ผู้ใช้ Guest (ไม่มี user_id): นับตาม IP Address ในไฟล์ JSON และรีเซ็ตอัตโนมัติทุกวัน
    """
    tier_upper = (user_tier or "FREE").upper()

    # ผู้ใช้ที่ Login แล้วจะไม่ถูกนับโควตาฝั่ง Backend เพื่อไม่ให้ IP quota ของ Guest มาตัดสิทธิ์
    if user_id:
        validate_tier_and_quota(user_tier=tier_upper, used_quota=0, model_type=model_type, pitch_shift_semitones=pitch_shift_semitones)
        return

    client_ip = request.client.host if request.client else "127.0.0.1"

    guest_data = _load_guest_quota()
    entry = guest_data.get(client_ip) or {}
    used_count = int(entry.get("count", 0))

    logger.info(f"[AUTH GUARD] IP: {client_ip} | Tier: {tier_upper} | Used: {used_count}/3")

    validate_tier_and_quota(user_tier=tier_upper, used_quota=used_count, model_type=model_type, pitch_shift_semitones=pitch_shift_semitones)

    if tier_upper == "FREE":
        entry["count"] = used_count + 1
        entry["date"] = date.today().isoformat()
        guest_data[client_ip] = entry
        _save_guest_quota(guest_data)
