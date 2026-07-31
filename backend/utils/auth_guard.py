import os
import json
import tempfile
import threading
import logging
from datetime import date
from fastapi import Request, HTTPException, status

logger = logging.getLogger("backend.auth_guard")

# ไฟล์ JSON สำหรับเก็บจำนวนการใช้งานของผู้ใช้ที่ไม่ได้ล็อกอิน (Guest) โดยใช้ IP Address
QUOTA_FILE = os.path.join(tempfile.gettempdir(), "harmoniq_guest_quota.json")

# ล็อกสำหรับอ่าน-แก้-เขียนไฟล์โควตา Guest แบบมีเธรดเดียว
# (กัน race condition เมื่อมี request พร้อมกันหลายตัว read-modify-write ชนกัน)
_guest_quota_lock = threading.Lock()


def _get_client_ip(request: Request) -> str:
    return request.client.host if request.client else "127.0.0.1"


def _load_guest_quota() -> dict[str, dict]:
    """
    อ่านข้อมูลโควตาของผู้ใช้ Guest จากไฟล์ JSON
    พร้อมรีเซ็ตโควตาที่หมดอายุรายวัน (อิงวันที่ในระบบ) และรองรับการย้าย format เก่า {"ip": count}
    ถ้าไฟล์เสียหายจะ log ไว้แทนที่จะเงียบๆ (B7)
    """
    if os.path.exists(QUOTA_FILE):
        try:
            with open(QUOTA_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                logger.warning(f"ไฟล์โควตา Guest ผิดรูปแบบ (ไม่ใช่ dict): {QUOTA_FILE} เริ่มนับใหม่")
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
        except Exception as e:
            logger.error(f"ไฟล์โควตา Guest เสียหาย อ่านไม่ได้: {e}")
    return {}


def _save_guest_quota(data: dict[str, dict]) -> None:
    """บันทึกไฟล์โควตาแบบ atomic: เขียนไฟล์ tmp ก่อนแล้วค่อย os.replace
    เพื่อไม่ให้ process ขาดกลางคันแล้วได้ไฟล์ JSON ครึ่งๆ กลางๆ (B7)"""
    try:
        tmp_path = QUOTA_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, QUOTA_FILE)
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


def validate_request_quota(request: Request, user_tier: str = "FREE", user_id: str | None = None, model_type: str = "LSTM", pitch_shift_semitones: int = 0):
    """
    ตรวจสอบสิทธิ์โดยไม่หักโควตา (B6: เรียกก่อนเริ่มงาน เพื่อให้ไฟล์ที่ไม่ผ่าน validate ไม่เสียโควตา)
    - ผู้ใช้ที่ Login แล้ว (user_id): ตรวจสิทธิ์ tier/โมเดล/pitch เท่านั้น (โควตานับฝั่ง Frontend ผ่าน DB)
    - ผู้ใช้ Guest (ไม่มี user_id): บังคับ tier=FREE เสมอ (B1: ไม่ trust header X-User-Tier)
      และตรวจจำนวนที่ใช้ตาม IP โดยไม่เพิ่มจำนวน
    """
    tier_upper = (user_tier or "FREE").upper()

    if user_id:
        validate_tier_and_quota(user_tier=tier_upper, used_quota=0, model_type=model_type, pitch_shift_semitones=pitch_shift_semitones)
        return

    client_ip = _get_client_ip(request)
    with _guest_quota_lock:
        guest_data = _load_guest_quota()
        entry = guest_data.get(client_ip) or {}
        used_count = int(entry.get("count", 0))

    logger.info(f"[AUTH GUARD] IP: {client_ip} | Tier: FREE (guest) | Used: {used_count}/3")

    validate_tier_and_quota(user_tier="FREE", used_quota=used_count, model_type=model_type, pitch_shift_semitones=pitch_shift_semitones)


def increment_guest_quota(request: Request, user_id: str | None = None) -> None:
    """
    นับโควตา Guest +1 หลังผ่านการตรวจสอบไฟล์ (save_upload) แล้วเท่านั้น (B6)
    ผู้ใช้ที่ Login แล้ว (user_id) จะไม่ถูกนับตาม IP เพราะโควตานับผ่าน Database ฝั่ง Frontend
    """
    if user_id:
        return

    client_ip = _get_client_ip(request)
    with _guest_quota_lock:
        guest_data = _load_guest_quota()
        entry = guest_data.get(client_ip) or {}
        entry["count"] = int(entry.get("count", 0)) + 1
        entry["date"] = date.today().isoformat()
        guest_data[client_ip] = entry
        _save_guest_quota(guest_data)
