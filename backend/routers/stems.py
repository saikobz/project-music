# backend/routers/stems.py
# FastAPI APIRouter สำหรับระบบแยกเสียงดนตรี (Stem Separation), ดาวน์โหลด และ Export

import os
import asyncio
import zipfile
import logging
import soundfile as sf
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, Header, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from backend.services.storage import save_upload, convert_to_mp3, processing_semaphore, UPLOAD_DIR
from backend.services.job_manager import job_manager
from backend.process_audio import separate_audio
from backend.auto_mastering import polish_vocal_file, apply_lufs_mastering
from backend.utils.auth_guard import validate_request_quota, increment_guest_quota

logger = logging.getLogger(__name__)
router = APIRouter(tags=["stems"])


def mixdown_stems(file_paths: list[str]) -> tuple[np.ndarray, int]:
    """รวมไฟล์เสียงหลายไฟล์เป็น mixdown ตัวเดียว (ใช้ร่วมกัน karaoke และ export mix — C2)

    - จัดการ mono/stereo ที่ปนกันโดยขยายเป็น stereo ให้อัตโนมัติ
    - ป้องกัน peak เกิน 1.0 (normalize)
    """
    mix = None
    samplerate = None

    for path in file_paths:
        data, sr = sf.read(path)
        if samplerate is None:
            samplerate = sr
        # ถ้าช่องสัญญาณไม่ตรงกัน ให้ขยายฝั่ง mono เป็น stereo (2 ช่อง)
        if mix is not None and data.ndim != mix.ndim:
            if data.ndim == 1:
                data = np.stack([data, data], axis=1)
            else:
                mix = np.stack([mix, mix], axis=1)
        if mix is None:
            mix = np.zeros_like(data)

        min_len = min(len(mix), len(data))
        mix[:min_len] += data[:min_len]

    if mix is None or samplerate is None:
        raise ValueError("ไม่มีไฟล์เสียงให้มิกซ์")

    max_val = np.max(np.abs(mix))
    if max_val > 1.0:
        mix = mix / max_val

    return mix.astype(np.float32), samplerate


def create_zip_archive(zip_path: str, entries: list[tuple[str, str]]) -> None:
    """สร้างไฟล์ zip จากรายการ (source_path, arcname) — ใช้ร่วมกัน /separate และ export (C3)"""
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file_path, arcname in entries:
            zipf.write(file_path, arcname)


@router.post("/separate")
async def separate(
    request: Request,
    file: UploadFile = File(...),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    x_user_tier: str = Header("FREE"),
    x_user_id: str = Header(None)
):
    """เอนด์พอยต์แยกแทร็กเสียง (Drums, Bass, Vocal, Other)"""
    # ตรวจสิทธิ์ก่อนเริ่มงาน (B6: ไฟล์ไม่ผ่าน validate จะไม่เสียโควตา)
    validate_request_quota(request=request, user_tier=x_user_tier, user_id=x_user_id)
    try:
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ผ่านการตรวจสอบไฟล์แล้ว -> ถึงค่อยนับโควตา (เฉพาะ Guest)
        increment_guest_quota(request, user_id=x_user_id)

        output_dir = os.path.join("separated", file_id)
        os.makedirs(output_dir, exist_ok=True)
        job_manager.register_job(file_id, output_dir)

        async with processing_semaphore:
            await asyncio.to_thread(separate_audio, input_path, output_dir)

        job_manager.complete_job(file_id)

        if export_format == "mp3":
            for root, _, files in os.walk(output_dir):
                for name in files:
                    if name.lower().endswith(".wav"):
                        wav_path = os.path.join(root, name)
                        # เก็บ WAV ต้นฉบับไว้ เพื่อให้ player/karaoke/vocal-polish ยังใช้งานได้
                        await asyncio.to_thread(convert_to_mp3, wav_path, remove_source=False)

        zip_filename = f"{file_id}_separated.zip"
        zip_path = os.path.join(UPLOAD_DIR, zip_filename)
        # สร้าง zip แบบ async (B10: เดิม block event loop กับไฟล์หลายร้อย MB)
        entries = []
        for root, _, files in os.walk(output_dir):
            for name in files:
                file_path = os.path.join(root, name)
                entries.append((file_path, os.path.relpath(file_path, output_dir)))
        await asyncio.to_thread(create_zip_archive, zip_path, entries)

        return JSONResponse(
            content={
                "status": "success",
                "file_id": file_id,
                "zip_url": f"/download/{file_id}",
            }
        )

    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@router.get("/download/{file_id}")
async def download_file(file_id: str):
    """ดาวน์โหลดไฟล์ผลลัพธ์ของ file_id ที่กำหนด (ZIP stems / Auto-EQ / Compressor / Pitch Shift)

    - ลำดับการค้นหา: ZIP ของ stems ใน uploads/ ก่อน -> แล้วค้นหาไฟล์ output ที่ขึ้นต้นด้วย
      {file_id}_ ใน uploads/, eq_applied/, compressed/ (ทุกไฟล์ถูก TTL cleanup ลบภายใน 20 นาที)
    - ถ้าไม่พบ -> 404 แสดงว่าไฟล์ถูกลบไปแล้ว (หมดอายุ) หรือไม่เคยถูกสร้าง
    """
    safe_file_id = os.path.basename(file_id)

    # 1) ZIP รวมทุก Stem (จาก /separate)
    zip_path = os.path.join(UPLOAD_DIR, f"{safe_file_id}_separated.zip")
    if os.path.exists(zip_path):
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="separated.zip",
        )

    # 2) ค้นหาไฟล์ output อื่น ๆ (Auto-EQ / Compressor / Pitch Shift) ที่ขึ้นต้นด้วย file_id_
    for directory in [UPLOAD_DIR, "eq_applied", "compressed"]:
        if not os.path.isdir(directory):
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in sorted(names):
            if name.startswith(f"{safe_file_id}_") and os.path.isfile(os.path.join(directory, name)):
                path = os.path.join(directory, name)
                ext = name.lower().rsplit(".", 1)[-1]
                media_type = (
                    "audio/mpeg" if ext == "mp3" else "audio/wav" if ext == "wav" else "application/octet-stream"
                )
                return FileResponse(path, media_type=media_type, filename=name)

    return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์สำหรับดาวน์โหลด (ไฟล์อาจถูกลบตามเวลาหมดอายุ)"})


@router.get("/separated/{file_id}/{filename}")
async def get_separated_file(file_id: str, filename: str):
    """ส่งคืนไฟล์เสียง Stem เดี่ยวจากโฟลเดอร์ผลลัพธ์"""
    safe_file_id = os.path.basename(file_id)
    safe_filename = os.path.basename(filename)
    folder = job_manager.get_job_directory(safe_file_id) or os.path.join("separated", safe_file_id)
    path = os.path.join(folder, safe_filename)

    if os.path.exists(path):
        ext = safe_filename.lower().split(".")[-1]
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "zip": "application/zip",
            "json": "application/json"
        }
        media_type = media_types.get(ext, "application/octet-stream")

        if ext == "zip":
            return FileResponse(
                path,
                media_type=media_type,
                filename=safe_filename
            )
        return FileResponse(path, media_type=media_type)
    return JSONResponse(status_code=404, content={"status": "error", "message": f"ไม่พบไฟล์ {filename}"})


@router.get("/karaoke/{file_id}")
async def get_karaoke(
    file_id: str,
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    """รวมไฟล์ Stem ดนตรี (Drums, Bass, Other) ทำเป็น Karaoke / Backing Track"""
    safe_file_id = os.path.basename(file_id)
    folder = job_manager.get_job_directory(safe_file_id) or os.path.join("separated", safe_file_id)
    karaoke_path = os.path.join(folder, "karaoke.wav")

    if os.path.exists(karaoke_path):
        return FileResponse(karaoke_path, media_type="audio/wav", filename="karaoke.wav")

    if not os.path.exists(folder):
        return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบข้อมูลการแยกเสียงสำหรับ file id นี้"})

    targets = ["drums.wav", "bass.wav", "other.wav"]

    try:
        existing_paths = [
            os.path.join(folder, target) for target in targets
            if os.path.exists(os.path.join(folder, target))
        ]
        if not existing_paths:
            return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ stem เสียงดนตรีเพื่อทำคาราโอเกะ"})

        mix, samplerate = mixdown_stems(existing_paths)
        sf.write(karaoke_path, mix, samplerate)

        if export_format == "mp3":
            karaoke_path = await asyncio.to_thread(convert_to_mp3, karaoke_path)
            return FileResponse(karaoke_path, media_type="audio/mpeg", filename="karaoke.mp3")

        return FileResponse(karaoke_path, media_type="audio/wav", filename="karaoke.wav")
    except ValueError:
        # mixdown_stems ไม่พบไฟล์ -> 404 (C6: exception อื่นปล่อยให้ global handler จัดการ)
        return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ stem เสียงดนตรีเพื่อทำคาราโอเกะ"})


@router.post("/api/process/vocal-polish")
async def process_vocal_polish(file_id: str = Query(...)):
    """API ขัดเกลาเสียงร้องอัตโนมัติ (Vocal Polish)"""
    safe_file_id = os.path.basename(file_id)
    folder = job_manager.get_job_directory(safe_file_id) or os.path.join("separated", safe_file_id)
    input_path = os.path.join(folder, "vocals.wav")
    
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์เสียงร้อง (vocals.wav) ในระบบ")
        
    output_filename = "vocals_polished.wav"
    output_path = os.path.join(folder, output_filename)
    
    try:
        async with processing_semaphore:
            await asyncio.to_thread(polish_vocal_file, input_path, output_path)
        return {"status": "success", "file_url": f"/separated/{safe_file_id}/{output_filename}"}
    except Exception as e:
        logger.error(f"Error polishing vocals: {e}")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการปรับแต่งเสียงร้อง")


@router.post("/api/process/export")
async def process_export(
    file_id: str = Query(...),
    export_type: str = Query("mix", pattern="^(mix|stems)$"),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    target_lufs: float = Query(-14.0),
    stems: list[str] = Query(...)
):
    """API สำหรับ Export Mixdown หรือ Stems แบบคัสตอม"""
    safe_file_id = os.path.basename(file_id)
    folder = job_manager.get_job_directory(safe_file_id) or os.path.join("separated", safe_file_id)
    
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลสำหรับการส่งออก")
        
    export_files = []
    
    try:
        valid_stems = ["vocals", "drums", "bass", "other"]
        selected_stem_files = []
        for stem in valid_stems:
            if stem in stems:
                filename = f"{stem}.wav"
                if stem == "vocals" and os.path.exists(os.path.join(folder, "vocals_polished.wav")):
                    filename = "vocals_polished.wav"
                
                path = os.path.join(folder, filename)
                if os.path.exists(path):
                    selected_stem_files.append((path, f"{stem}.wav"))
                    
        if not selected_stem_files:
            raise HTTPException(status_code=400, detail="กรุณาเลือกอย่างน้อย 1 แทร็กเพื่อ Export")
            
        if export_type == "mix":
            mix, samplerate = mixdown_stems([path for path, _ in selected_stem_files])

            mixed_path = os.path.join(folder, "mixed_custom.wav")
            sf.write(mixed_path, mix, samplerate)

            output_filename = f"custom_mix_{target_lufs}.wav"
            output_path = os.path.join(folder, output_filename)

            async with processing_semaphore:
                await asyncio.to_thread(apply_lufs_mastering, mixed_path, output_path, target_lufs)

            if export_format == "mp3":
                output_path = await asyncio.to_thread(convert_to_mp3, output_path)
                output_filename = os.path.basename(output_path)

            export_files.append((output_path, output_filename))
        else:
            for path, arcname in selected_stem_files:
                if export_format == "mp3":
                    # เก็บ WAV ต้นฉบับไว้ เพื่อไม่ให้ stem หายก่อน TTL และ export ซ้ำได้
                    path = await asyncio.to_thread(convert_to_mp3, path, remove_source=False)
                    arcname = arcname.replace(".wav", ".mp3")
                export_files.append((path, arcname))
                
        if len(export_files) == 1:
            file_path, arcname = export_files[0]
            filename_only = os.path.basename(file_path)
            return {
                "status": "success", 
                "type": "file",
                "file_url": f"/separated/{safe_file_id}/{filename_only}",
                "filename": arcname
            }
        else:
            zip_filename = f"export_stems_{export_format}.zip"
            zip_path = os.path.join(folder, zip_filename)
            await asyncio.to_thread(create_zip_archive, zip_path, export_files)
            return {
                "status": "success",
                "type": "zip",
                "file_url": f"/separated/{safe_file_id}/{zip_filename}",
                "filename": f"HarmoniQ_Stems_{safe_file_id[:6]}.zip"
            }
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error exporting audio: {e}")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในขั้นตอน Export")
