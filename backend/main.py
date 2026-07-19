# - เป็นศูนย์กลางของ FastAPI backend และประกาศ API routes ทั้งหมดของระบบ
# - รับไฟล์อัปโหลด ตรวจสอบชนิด/ขนาดไฟล์ แล้วส่งงานไปยังโมดูลประมวลผลเสียงแต่ละตัว
# - จัดการเรื่อง response, error handling และ cleanup ไฟล์ชั่วคราวหลังประมวลผล

import os
import asyncio
import shutil
import threading
import zipfile
import logging
import soundfile as sf
import numpy as np
from uuid import uuid4
from typing import Tuple

# ชุด import ของ FastAPI สำหรับสร้างแอป, รับไฟล์, และส่ง response กลับ
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

from backend.cleanup_task import periodic_cleanup

# import ฟังก์ชันประมวลผลจากโมดูลย่อยของ backend
from backend.process_audio import separate_audio, analyze_audio, pitch_shift_audio
from backend.eq_compressor import apply_compression
from backend.auto_eq_inference import (
    apply_auto_eq_file,
    AutoEQModelLoadError,
    DELTA_CLAMP_DB,
    MIN_DELTA_CLAMP_DB,
    MAX_DELTA_CLAMP_DB,
    DEFAULT_AUTO_EQ_MODEL_ID,
    SUPPORTED_AUTO_EQ_MODELS,
)
from backend.auto_mastering import polish_vocal_file, apply_lufs_mastering

# ไฟล์นี้เป็นจุดรวมการตั้งค่า FastAPI และประกาศ endpoint หลักของระบบทั้งหมด

# เวลาหลังประมวลผลเสร็จที่ไฟล์ชั่วคราวจะถูกลบทิ้งอัตโนมัติ
cleanup_ttl = int(os.getenv("SEPARATE_TTL_SECONDS", "1200"))  # 20 นาทีลบ

@asynccontextmanager
async def lifespan(app: FastAPI):
    # เริ่มต้น background task กวาดลบไฟล์ขยะ
    cleanup_task = asyncio.create_task(periodic_cleanup(interval_seconds=300, ttl_seconds=cleanup_ttl))
    yield
    # เมื่อเซิร์ฟเวอร์ปิดการทำงาน
    cleanup_task.cancel()

# การตั้งค่าแอปพลิเคชัน
app = FastAPI(lifespan=lifespan)
# logger ใช้บันทึก error หรือสถานะสำคัญของ backend ระหว่างรันจริง
logger = logging.getLogger(__name__)

# จำกัดจำนวนการประมวลผล AI พร้อมกันเพื่อป้องกัน CPU/RAM เต็ม
MAX_CONCURRENT_TASKS = 2
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# โฟลเดอร์ทำงานและข้อจำกัดของไฟล์
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ค่าคอนฟิกระหว่างรัน
# อ่าน origin ที่อนุญาตจาก environment เพื่อให้เปลี่ยนได้โดยไม่ต้องแก้โค้ด
allow_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
# แยกค่า origin ที่คั่นด้วย comma ให้เป็น list สำหรับ CORS middleware
allow_origins = [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]

# จำกัดขนาดไฟล์อัปโหลดสูงสุดเพื่อกันใช้หน่วยความจำมากเกินไป
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB

# เปิด CORS ให้ frontend ที่กำหนดสามารถเรียก API นี้ได้จาก browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health():
    # health check แบบง่าย ใช้ตรวจว่า backend ยังตอบสนองได้ตามปกติ
    return {"status": "ok"}


# ตั้งค่าอัปโหลดไฟล์
async def save_upload(
    file: UploadFile,
    upload_dir: str = UPLOAD_DIR,
    trim_start: float | None = None,
    trim_end: float | None = None
) -> Tuple[str, str]:
    # ฟังก์ชันช่วยกลางสำหรับ endpoint ที่ต้องรับไฟล์เสียงเข้ามา
    # งานของมันคือ validate ไฟล์, สร้างชื่อไม่ซ้ำ, และเขียนไฟล์ลงดิสก์
    # ดึงชื่อไฟล์เดิมจาก request; ถ้าไม่มีชื่อไฟล์ให้ใช้สตริงว่างแทน
    filename = os.path.basename(file.filename or "")

    # แยกนามสกุลไฟล์ออกมาเพื่อตรวจชนิดไฟล์
    _, ext = os.path.splitext(filename)

    # backend นี้รับเฉพาะไฟล์ .wav เท่านั้น ถ้าไม่ใช่ให้ตอบกลับ 400 ทันที
    if ext.lower() != ".wav":
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ WAV (.wav)")

    # ป้องกันการอัปโหลดไฟล์ใหญ่เกินที่ระบบกำหนด (ตรวจจาก Header ก่อนถ้ามี)
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    # สร้าง id ไม่ซ้ำเพื่อกันชื่อไฟล์ชนกันและใช้อ้างอิงไฟล์นี้ในขั้นตอนถัดไป
    file_id = str(uuid4())

    # รวม file_id กับชื่อไฟล์เดิมเพื่อให้ได้ชื่อไฟล์ที่เก็บจริงบนดิสก์
    stored_name = f"{file_id}_{filename}"

    # สร้าง path เต็มของไฟล์ที่จะถูกบันทึกลงในโฟลเดอร์ uploads
    input_path = os.path.join(upload_dir, stored_name)

    # เขียนไฟล์ลงดิสก์แบบสตรีมมิ่งผ่าน shutil.copyfileobj ลดปัญหาการกิน RAM 100% ตอนอัปโหลดไฟล์ใหญ่
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # เช็คขนาดไฟล์อีกครั้งหลังเขียนเสร็จ (ป้องกันกรณีขนาดไม่ส่งมาใน Header)
    if os.path.getsize(input_path) > MAX_UPLOAD_BYTES:
        os.remove(input_path)
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    # ตัดช่วงเสียงถ้ามีการกำหนดช่วงเวลา
    if trim_start is not None or trim_end is not None:
        try:
            audio_data, samplerate = sf.read(input_path)
            start_frame = int(trim_start * samplerate) if trim_start is not None else 0
            end_frame = int(trim_end * samplerate) if trim_end is not None else len(audio_data)
            trimmed_data = audio_data[start_frame:end_frame]
            sf.write(input_path, trimmed_data, samplerate)
        except Exception as e:
            logger.error(f"Error trimming audio: {e}")
            raise HTTPException(status_code=400, detail=f"การตัดช่วงเวลาเสียงล้มเหลว: {e}")

    # คืนทั้ง file_id และ path ของไฟล์ที่บันทึกแล้วให้ endpoint ที่เรียกใช้เอาไปทำงานต่อ
    return file_id, input_path





def convert_to_mp3(wav_path: str) -> str:
    """แปลงไฟล์ wav เป็น mp3 และลบไฟล์ wav ทิ้ง คืนค่าเป็นพาท mp3"""
    try:
        from pydub import AudioSegment
        mp3_path = wav_path.rsplit(".", 1)[0] + ".mp3"
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate="320k")
        if os.path.exists(wav_path):
            os.remove(wav_path)
        return mp3_path
    except Exception as e:
        logger.error(f"Error converting to mp3: {e}")
        return wav_path


#เส้น api แยกเสียง
@app.post("/separate")
async def separate(
    file: UploadFile = File(...),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    try:
        # บันทึกไฟล์ที่อัปโหลดก่อน แล้วค่อยส่ง path ให้โมดูลแยก stem ทำงานต่อ
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)

        # โฟลเดอร์ผลลัพธ์จะแยกตาม file_id เพื่อไม่ให้แต่ละงานชนกัน
        output_dir = os.path.join("separated", file_id)
        os.makedirs(output_dir, exist_ok=True)
        # รันงานแยก stem ใน thread แยก โดยรอคิวผ่าน semaphore เพื่อไม่ให้ CPU โหลดหนักเกินไป
        async with processing_semaphore:
            await asyncio.to_thread(separate_audio, input_path, output_dir)

        # แปลงเป็น MP3 ถ้าผู้ใช้เลือก
        if export_format == "mp3":
            for root, _, files in os.walk(output_dir):
                for name in files:
                    if name.lower().endswith(".wav"):
                        wav_path = os.path.join(root, name)
                        await asyncio.to_thread(convert_to_mp3, wav_path)

        # รวม stem ทั้งหมดเป็น ZIP เพื่อให้ frontend ดาวน์โหลดทีเดียวได้
        zip_filename = f"{file_id}_separated.zip"
        zip_path = os.path.join(UPLOAD_DIR, zip_filename)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            # เดินทุกไฟล์ใน output_dir แล้วเพิ่มเข้า zip โดยคง path ภายในแบบสัมพัทธ์
            for root, _, files in os.walk(output_dir):
                for name in files:
                    file_path = os.path.join(root, name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        # ลบไฟล์อัปโหลดต้นฉบับทันทีเพราะไม่ต้องใช้ต่อแล้ว
        if os.path.exists(input_path):
            os.remove(input_path)

        # ส่ง URL ที่ frontend ใช้เรียกดาวน์โหลด zip ภายหลัง
        return JSONResponse(
            content={
                "status": "success",
                "file_id": file_id,
                "zip_url": f"/download/{file_id}",
            }
        )

    except HTTPException as http_exc:
        # ถ้าเป็น error ที่ตั้งใจโยนไว้แล้ว เช่น 400 ให้ส่งต่อออกไปตรง ๆ
        raise http_exc
    except Exception as e:
        # error ที่ไม่คาดคิดจะถูกรวมเป็น 500
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/download/{file_id}")
async def download_zip(file_id: str):
    # ใช้ file_id ที่ frontend ส่งมาเพื่อหา ZIP ที่ถูกสร้างจากงานแยก stem
    safe_file_id = os.path.basename(file_id)
    zip_filename = f"{safe_file_id}_separated.zip"
    zip_path = os.path.join(UPLOAD_DIR, zip_filename)

    if os.path.exists(zip_path):
        # ถ้าเจอไฟล์ zip ก็สตรีมกลับไปเป็นไฟล์ดาวน์โหลด
        return FileResponse(
            zip_path,
            # ระบุ type ว่าไฟล์ที่ส่งกลับเป็น zip เพื่อให้ browser/frontend จัดการดาวน์โหลดได้ถูกชนิด
            media_type="application/zip",
            filename="separated.zip",
        )
    return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ zip สำหรับดาวน์โหลด"})


@app.get("/separated/{file_id}/{filename}")
async def get_separated_file(file_id: str, filename: str):
    # ส่งคืนไฟล์ใด ๆ ในโฟลเดอร์ separated (เช่น wav, mp3, zip)
    safe_file_id = os.path.basename(file_id)
    safe_filename = os.path.basename(filename)
    folder = os.path.join("separated", safe_file_id)
    path = os.path.join(folder, safe_filename)

    if os.path.exists(path):
        # ตรวจสอบนามสกุลไฟล์เพื่อกำหนด media type และ headers ที่ถูกต้อง
        ext = safe_filename.lower().split(".")[-1]
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "zip": "application/zip",
            "json": "application/json"
        }
        media_type = media_types.get(ext, "application/octet-stream")
        
        # สำหรับไฟล์ zip หรือเมื่อต้องการบังคับดาวน์โหลด
        if ext == "zip":
            return FileResponse(
                path,
                media_type=media_type,
                filename=safe_filename
            )
        return FileResponse(path, media_type=media_type)
    return JSONResponse(status_code=404, content={"status": "error", "message": f"ไม่พบไฟล์ {filename}"})



@app.get("/karaoke/{file_id}")
async def get_karaoke(
    file_id: str,
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    # รวมไฟล์ stem ดนตรี (Drums, Bass, Other) เข้าด้วยกันเพื่อทำ Karaoke / Backing Track
    safe_file_id = os.path.basename(file_id)
    folder = os.path.join("separated", safe_file_id)
    karaoke_path = os.path.join(folder, "karaoke.wav")

    # ถ้าสร้างไฟล์ไว้แล้ว โหลดได้เลยไม่ต้องสร้างใหม่
    if os.path.exists(karaoke_path):
        return FileResponse(karaoke_path, media_type="audio/wav", filename="karaoke.wav")

    if not os.path.exists(folder):
        return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบข้อมูลการแยกเสียงสำหรับ file id นี้"})

    # stems ที่จะรวม
    targets = ["drums.wav", "bass.wav", "other.wav"]
    mix = None
    samplerate = None

    try:
        for target in targets:
            path = os.path.join(folder, target)
            if os.path.exists(path):
                data, sr = sf.read(path)
                if samplerate is None:
                    samplerate = sr
                if mix is None:
                    mix = np.zeros_like(data)
                
                # เอาความยาวที่สั้นที่สุด เพื่อกันขนาด array ไม่เท่ากัน
                min_len = min(len(mix), len(data))
                mix[:min_len] += data[:min_len]

        if mix is None:
            return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ stem เสียงดนตรีเพื่อทำคาราโอเกะ"})

        # กันค่าเกิน (Clip) ถ้าเกิดการรวมเสียงแล้วดังเกินไป
        max_val = np.max(np.abs(mix))
        if max_val > 1.0:
            mix = mix / max_val

        # export 
        sf.write(karaoke_path, mix, samplerate)

        if export_format == "mp3":
            karaoke_path = await asyncio.to_thread(convert_to_mp3, karaoke_path)
            return FileResponse(karaoke_path, media_type="audio/mpeg", filename="karaoke.mp3")

        return FileResponse(karaoke_path, media_type="audio/wav", filename="karaoke.wav")
    except Exception as e:
        logger.error(f"Error creating karaoke mixdown: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"การรวมไฟล์คาราโอเกะล้มเหลว: {str(e)}"})


# เส้นทาง Auto-EQ แบบ AI
@app.post("/apply-eq-ai")
async def apply_eq_ai(
    file: UploadFile = File(...),
    genre: str = Query("pop", description="แนวเพลง เช่น pop, rock, trap, country, soul"),
    model_id: str = Query(
        DEFAULT_AUTO_EQ_MODEL_ID,
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
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    try:
        # บันทึกไฟล์ก่อน แล้วส่ง path ไปให้โมเดล Auto-EQ ประมวลผล
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ตั้งชื่อไฟล์ผลลัพธ์ให้สะท้อนว่าเป็นงาน EQ AI และใช้ genre อะไร
        output_filename = f"{file_id}_eq_ai_{model_id}_{genre}.wav"
        output_path = os.path.join("eq_applied", output_filename)
        # inference รันใน worker thread โดยรอคิวผ่าน semaphore เพื่อไม่ให้ CPU โหลดหนักเกินไป
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
        # แยกกรณีโมเดลโหลดไม่ได้ออกมาเป็น 503 เพื่อบอกว่าบริการยังไม่พร้อม
        logger.exception("Auto-EQ model unavailable: %s", model_exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error_code": "AUTO_EQ_MODEL_UNAVAILABLE",
                "message": str(model_exc),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # ไม่ว่าผลจะสำเร็จหรือ error ให้ลบไฟล์อัปโหลดต้นฉบับเสมอ
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


# api compressor
@app.post("/apply-compressor")
async def apply_compressor(
    file: UploadFile = File(...),
    # strength เป็น preset ความแรงแบบง่ายสำหรับผู้ใช้ทั่วไป
    strength: str = Query("medium", pattern="^(soft|medium|hard)$"),
    # genre ใช้เลือก preset เริ่มต้นของ compressor ให้เข้ากับแนวเพลง
    genre: str = Query("general", description="general, pop, rock, trap, country, soul"),
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
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    try:
        # งาน compressor ใช้ทั้ง preset และค่าที่ผู้ใช้ override ผ่าน query string
        _, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # เรียกใช้ฟังก์ชัน DSP ใน thread แยก โดยรอคิวผ่าน semaphore
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
        # ถ้าค่าพารามิเตอร์ไม่สมเหตุผล ให้ตอบกลับเป็น 400 พร้อมข้อความอธิบาย
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # ลบไฟล์อัปโหลดต้นทางทุกครั้งหลังประมวลผลเสร็จ
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)

# เส้นทางปรับ pitch
@app.post("/pitch-shift")
async def pitch_shift(
    file: UploadFile = File(...), 
    steps: float = 0,
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None),
    export_format: str = Query("wav", pattern="^(wav|mp3)$")
):
    try:
        # บันทึกไฟล์ก่อน จากนั้นสร้างไฟล์ผลลัพธ์ที่ถูก shift pitch แล้วส่งกลับ
        file_id, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # ตั้งชื่อไฟล์ผลลัพธ์ใหม่เพื่อไม่เขียนทับต้นฉบับ
        output_filename = f"{file_id}_pitch.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        # ฟังก์ชัน pitch_shift_audio จะคืน path ของไฟล์ผลลัพธ์กลับมา
        # ใช้ to_thread โดยรอคิวผ่าน semaphore
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
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # ลบไฟล์ต้นฉบับหลังใช้งานเสร็จเสมอ
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


# เส้น วิเคราะห์เสียง
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    trim_start: float | None = Query(None),
    trim_end: float | None = Query(None)
):
    try:
        # endpoint นี้คืนข้อมูลวิเคราะห์เป็น JSON ไม่ได้สร้างไฟล์เสียงใหม่
        _, input_path = await save_upload(file, trim_start=trim_start, trim_end=trim_end)
        # วิเคราะห์เสียงใน thread แยก โดยรอคิวผ่าน semaphore
        async with processing_semaphore:
            result = await asyncio.to_thread(analyze_audio, input_path)
        return JSONResponse(content=result)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        # งาน analyze ใช้ไฟล์ชั่วคราวระยะสั้น จึงลบทิ้งทันทีหลังได้ผล
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/api/process/vocal-polish")
async def process_vocal_polish(file_id: str = Query(...)):
    """API สำหรับขัดเกลาเสียงร้องอัตโนมัติ"""
    safe_file_id = os.path.basename(file_id)
    folder = os.path.join("separated", safe_file_id)
    input_path = os.path.join(folder, "vocals.wav")
    
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์เสียงร้อง (vocals.wav) ในระบบ")
        
    output_filename = "vocals_polished.wav"
    output_path = os.path.join(folder, output_filename)
    
    try:
        async with processing_semaphore:
            await asyncio.to_thread(polish_vocal_file, input_path, output_path)
        # เราส่งกลับเป็น URL เดียวกันกับที่ใช้ดึง stem แต่เปลี่ยนชื่อไฟล์
        return {"status": "success", "file_url": f"/separated/{safe_file_id}/{output_filename}"}
    except Exception as e:
        logger.error(f"Error polishing vocals: {e}")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการปรับแต่งเสียงร้อง")

@app.post("/api/process/export")
async def process_export(
    file_id: str = Query(...),
    export_type: str = Query("mix", pattern="^(mix|stems)$"),
    export_format: str = Query("wav", pattern="^(wav|mp3)$"),
    target_lufs: float = Query(-14.0),
    stems: list[str] = Query(...)
):
    """API สำหรับ Export Mixdown และ/หรือ Stems แยกตามความต้องการ"""
    safe_file_id = os.path.basename(file_id)
    folder = os.path.join("separated", safe_file_id)
    
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="ไม่พบข้อมูลสำหรับการส่งออก")
        
    export_files = [] # list of (file_path, arcname)
    
    try:
        # 1. คัดกรอง Stems ที่ผู้ใช้ต้องการ
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
            # 2. นำ Stems ที่เลือกมาผสมกัน (Mix)
            mix = None
            samplerate = None
            
            for path, _ in selected_stem_files:
                data, sr = sf.read(path)
                if samplerate is None:
                    samplerate = sr
                if mix is None:
                    mix = np.zeros_like(data)
                min_len = min(len(mix), len(data))
                mix[:min_len] += data[:min_len]
                
            if mix is not None:
                max_val = np.max(np.abs(mix))
                if max_val > 1.0:
                    mix = mix / max_val
                
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
            # export_type == "stems"
            # 3. เตรียมไฟล์สำหรับ ZIP
            for path, arcname in selected_stem_files:
                if export_format == "mp3":
                    path = await asyncio.to_thread(convert_to_mp3, path)
                    arcname = arcname.replace(".wav", ".mp3")
                export_files.append((path, arcname))
                
        # 4. ตรวจสอบจำนวนไฟล์ที่ส่งออก
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
            
            def create_zip():
                with zipfile.ZipFile(zip_path, "w") as zipf:
                    for f_path, a_name in export_files:
                        zipf.write(f_path, a_name)
                        
            await asyncio.to_thread(create_zip)
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
