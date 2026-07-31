# backend/routers/audio_ops.py
# FastAPI APIRouter สำหรับระบบ Auto-EQ, Compressor, Pitch Shift และ Audio Analysis

import os
import asyncio
import logging
import shutil
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, Query, Header, Request, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse

from backend.services.storage import save_upload, convert_to_mp3, processing_semaphore, UPLOAD_DIR
from backend.process_audio import analyze_audio, pitch_shift_audio
from backend.eq_compressor import apply_compression
from backend.utils.auth_guard import validate_request_quota, increment_guest_quota
from backend.auto_eq_inference import (
    apply_auto_eq_file,
    AutoEQModelLoadError,
    DELTA_CLAMP_DB,
    MIN_DELTA_CLAMP_DB,
    MAX_DELTA_CLAMP_DB,
    DEFAULT_AUTO_EQ_MODEL_ID,
    SUPPORTED_AUTO_EQ_MODELS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["audio_ops"])


@router.post("/apply-eq-ai")
async def apply_eq_ai(
    request: Request,
    file: UploadFile = File(...),
    genre: str = Query(
        "pop",
        pattern="^(pop|rock|trap|country|soul)$",
        description="แนวเพลง เช่น pop, rock, trap, country, soul",
    ),
    model_id: str = Query(
        DEFAULT_AUTO_EQ_MODEL_ID,
        pattern="^(cnn-v1|lstm-last)$",
        description=f"Auto-EQ model id: {', '.join(SUPPORTED_AUTO_EQ_MODELS)}",
    ),
    delta_clamp_db: float = Query(
        DELTA_CLAMP_DB,
        ge=MIN_DELTA_CLAMP_DB,
        le=MAX_DELTA_CLAMP_DB,
        description="เพดานการปรับ EQ ต่อจุดในหน่วย dB",
    ),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    x_user_tier: str = Header("FREE"),
    x_user_id: str = Header(None)
):
    """เอนด์พอยต์ปรับแต่ง EQ อัตโนมัติด้วย AI"""
    # ตรวจสิทธิ์ก่อนเริ่มงาน (B6: ไฟล์ไม่ผ่าน validate จะไม่เสียโควตา)
    validate_request_quota(request=request, user_tier=x_user_tier, user_id=x_user_id, model_type=model_id)
    try:
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ผ่านการตรวจสอบไฟล์แล้ว -> ถึงค่อยนับโควตา (เฉพาะ Guest)
        increment_guest_quota(request, user_id=x_user_id)
        output_filename = f"{file_id}_eq_ai_{model_id}_{genre}.wav"
        output_path = os.path.join("eq_applied", output_filename)
        os.makedirs("eq_applied", exist_ok=True)

        async with processing_semaphore:
            result_path = await asyncio.to_thread(
                apply_auto_eq_file,
                input_path,
                output_path,
                genre,
                delta_clamp_db,
                model_id,
            )

        if export_format == "mp3":
            result_path = await asyncio.to_thread(convert_to_mp3, result_path)

        return FileResponse(
            result_path,
            media_type="audio/mpeg" if export_format == "mp3" else "audio/wav",
            filename=os.path.basename(result_path),
        )
    except HTTPException as http_exc:
        raise http_exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except AutoEQModelLoadError as model_exc:
        logger.exception("Auto-EQ model unavailable: %s", model_exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error_code": "AUTO_EQ_MODEL_UNAVAILABLE",
                "message": str(model_exc),
            },
        )
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@router.post("/apply-compressor")
async def apply_compressor(
    request: Request,
    file: UploadFile = File(...),
    strength: str = Query("medium", pattern="^(soft|medium|hard)$"),
    genre: str = Query(
        "general",
        pattern="^(general|pop|rock|trap|country|soul)$",
        description="general, pop, rock, trap, country, soul",
    ),
    threshold: float | None = Query(None, ge=-80.0, le=0.0, description="dBFS"),
    ratio: float | None = Query(None, ge=1.0, le=20.0),
    attack: float | None = Query(None, ge=0.1, le=200.0, description="ms"),
    release: float | None = Query(None, ge=0.1, le=1000.0, description="ms"),
    knee: float | None = Query(None, ge=0.0, le=24.0, description="dB"),
    makeup_gain: float = Query(0.0, ge=-24.0, le=24.0, description="dB"),
    dry_wet: float = Query(100.0, ge=0.0, le=100.0, description="percent"),
    output_ceiling: float | None = Query(None, ge=-20.0, le=0.0, description="dBFS"),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    x_user_tier: str = Header("FREE"),
    x_user_id: str = Header(None)
):
    """เอนด์พอยต์ปรับแต่ง Compressor เสียง"""
    # ตรวจสิทธิ์ก่อนเริ่มงาน (B6: ไฟล์ไม่ผ่าน validate จะไม่เสียโควตา)
    validate_request_quota(request=request, user_tier=x_user_tier, user_id=x_user_id)
    try:
        _, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ผ่านการตรวจสอบไฟล์แล้ว -> ถึงค่อยนับโควตา (เฉพาะ Guest)
        increment_guest_quota(request, user_id=x_user_id)
        os.makedirs("compressed", exist_ok=True)

        async with processing_semaphore:
            output_path = await asyncio.to_thread(
                apply_compression,
                input_path,
                strength,
                genre,
                "compressed",
                threshold=threshold,
                ratio=ratio,
                attack=attack,
                release=release,
                knee=knee,
                makeup_gain=makeup_gain,
                dry_wet=dry_wet,
                output_ceiling=output_ceiling,
            )

        if export_format == "mp3":
            output_path = await asyncio.to_thread(convert_to_mp3, output_path)

        return FileResponse(
            output_path,
            media_type="audio/mpeg" if export_format == "mp3" else "audio/wav",
            filename=os.path.basename(output_path),
        )
    except HTTPException as http_exc:
        raise http_exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@router.post("/pitch-shift")
async def pitch_shift(
    request: Request,
    file: UploadFile = File(...), 
    steps: float = 0,
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    x_user_tier: str = Header("FREE"),
    x_user_id: str = Header(None)
):
    """เอนด์พอยต์ปรับ Pitch ของไฟล์เสียง"""
    # ตรวจสิทธิ์ก่อนเริ่มงาน (B6: ไฟล์ไม่ผ่าน validate จะไม่เสียโควตา)
    validate_request_quota(request=request, user_tier=x_user_tier, user_id=x_user_id, pitch_shift_semitones=int(steps))
    try:
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ผ่านการตรวจสอบไฟล์แล้ว -> ถึงค่อยนับโควตา (เฉพาะ Guest)
        increment_guest_quota(request, user_id=x_user_id)
        output_filename = f"{file_id}_pitch.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        async with processing_semaphore:
            result_path = await asyncio.to_thread(pitch_shift_audio, input_path, steps, output_path)

        if export_format == "mp3":
            result_path = await asyncio.to_thread(convert_to_mp3, result_path)

        return FileResponse(
            result_path,
            media_type="audio/mpeg" if export_format == "mp3" else "audio/wav",
            filename=os.path.basename(result_path),
        )
    except HTTPException as http_exc:
        raise http_exc
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@router.post("/convert-format")
async def convert_format(
    file: UploadFile = File(...),
    export_format: str = Query("mp3", pattern="^(wav|mp3)$"),
    background: BackgroundTasks = BackgroundTasks(),
):
    """แปลงไฟล์เสียงที่ประมวลผลแล้ว (WAV/MP3) เป็น format อื่น

    ใช้กับ SingleExportModal เมื่อผู้ใช้เปลี่ยน format หลังประมวลผลเสร็จ
    - ไม่ต้องประมวลผลใหม่ (AI/DSP) และไม่หักโควตา (F12)
    - ไฟล์ชั่วคราวถูกลบหลังส่ง response ผ่าน BackgroundTasks
    """
    filename = os.path.basename(file.filename or "audio.wav")
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    if ext not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ WAV หรือ MP3")

    file_id = str(uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if os.path.getsize(input_path) > 100 * 1024 * 1024:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    try:
        result_path = input_path
        if ext != export_format:
            from pydub import AudioSegment
            output_path = os.path.join(UPLOAD_DIR, f"{file_id}.{export_format}")
            audio = AudioSegment.from_file(input_path, format=ext)
            if export_format == "mp3":
                audio.export(output_path, format="mp3", bitrate="320k")
            else:
                audio.export(output_path, format="wav")
            result_path = output_path
            background.add_task(_cleanup_convert_files, input_path)

        download_name = f"{os.path.splitext(filename)[0]}.{export_format}"
        return FileResponse(
            result_path,
            media_type="audio/mpeg" if export_format == "mp3" else "audio/wav",
            filename=download_name,
            background=background,
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error converting format: {e}")
        raise HTTPException(status_code=500, detail="การแปลงฟอร์แมตเสียงล้มเหลว")


def _cleanup_convert_files(path: str) -> None:
    """ลบไฟล์ชั่วคราวหลัง response ถูกส่งเสร็จ"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None)
):
    """เอนด์พอยต์วิเคราะห์ค่าสเปกตรัมและความถี่เสียง"""
    try:
        _, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        async with processing_semaphore:
            result = await asyncio.to_thread(analyze_audio, input_path)
        return JSONResponse(content=result)
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
