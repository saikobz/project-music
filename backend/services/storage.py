# backend/services/storage.py
# โมดูลกลางจัดการไฟล์อัปโหลด การตรวจสอบขนาด/นามสกุลไฟล์ การแปลงฟอร์แมต และการตัดช่วงเสียง

import os
import asyncio
import shutil
import logging
import soundfile as sf
from uuid import uuid4
from typing import Tuple
from fastapi import UploadFile, HTTPException

from backend.config import MAX_UPLOAD_BYTES, DIR_UPLOADS, MAX_CONCURRENT_TASKS

logger = logging.getLogger(__name__)

UPLOAD_DIR = DIR_UPLOADS
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Semaphore สำหรับจำกัด Concurrency การประมวลผล AI/DSP
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


async def save_upload(
    file: UploadFile,
    upload_dir: str = UPLOAD_DIR,
    trim_start: float | None = None,
    trim_end: float | None = None
) -> Tuple[str, str]:
    """บันทึกไฟล์อัปโหลดลงดิสก์ ตรวจสอบความถูกต้อง และตัดช่วงเวลาเสียงถ้ามีการระบุ"""
    filename = os.path.basename(file.filename or "")
    _, ext = os.path.splitext(filename)

    if ext.lower() != ".wav":
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ WAV (.wav)")

    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    file_id = str(uuid4())
    stored_name = f"{file_id}_{filename}"
    input_path = os.path.join(upload_dir, stored_name)

    # สตรีมมิ่งเขียนไฟล์ลงดิสก์เพื่อประหยัดหน่วยความจำ
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if os.path.getsize(input_path) > MAX_UPLOAD_BYTES:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    # ตัดช่วงเสียงถ้ามี trim_start หรือ trim_end
    if trim_start is not None or trim_end is not None:
        try:
            audio_data, samplerate = sf.read(input_path)
            duration = len(audio_data) / float(samplerate)

            start = float(trim_start) if trim_start is not None else 0.0
            end = float(trim_end) if trim_end is not None else duration

            # ตรวจสอบช่วงเวลาให้ถูกต้องก่อนตัด เพื่อป้องกัน slice ติดลบหรือไฟล์ว่าง
            if start < 0.0:
                raise HTTPException(status_code=400, detail="trim_start ต้องไม่น้อยกว่า 0 วินาที")
            if start >= duration:
                raise HTTPException(status_code=400, detail="trim_start ต้องน้อยกว่าความยาวไฟล์")
            if end <= start:
                raise HTTPException(status_code=400, detail="trim_end ต้องมากกว่า trim_start")
            if end > duration:
                raise HTTPException(status_code=400, detail="trim_end ต้องไม่เกินความยาวไฟล์")

            start_frame = int(start * samplerate)
            end_frame = int(end * samplerate)
            trimmed_data = audio_data[start_frame:end_frame]
            if len(trimmed_data) == 0:
                raise HTTPException(status_code=400, detail="ช่วงเวลาที่เลือกไม่มีความยาวเสียง")
            sf.write(input_path, trimmed_data, samplerate)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error trimming audio: {e}")
            raise HTTPException(status_code=400, detail=f"การตัดช่วงเวลาเสียงล้มเหลว: {e}")

    return file_id, input_path


def convert_to_mp3(wav_path: str, remove_source: bool = True) -> str:
    """แปลงไฟล์ WAV เป็น MP3

    - remove_source=True: ลบไฟล์ WAV ต้นทางหลังแปลงเสร็จ (ใช้กับไฟล์ผลลัพธ์ที่สร้างใหม่ เช่น karaoke/mix)
    - remove_source=False: เก็บไฟล์ WAV ต้นฉบับไว้ (ใช้กับ Stem ต้นฉบับเพื่อไม่ให้ไฟล์หายก่อน TTL)
    - ถ้าแปลงไม่สำเร็จจะ raise RuntimeError แทนการคืน path เดิมแบบเงียบๆ
    """
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("ต้องติดตั้ง pydub ก่อน: pip install pydub") from exc

    try:
        mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="320k")
        if remove_source and os.path.exists(wav_path):
            os.remove(wav_path)
        return mp3_path
    except Exception as e:
        logger.error(f"Error converting to mp3: {e}")
        raise RuntimeError(f"ไม่สามารถแปลงไฟล์เป็น MP3 ได้: {e}") from e
