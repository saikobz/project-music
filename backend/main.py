# - เป็นศูนย์กลางของ FastAPI backend และประกาศ API routes ทั้งหมดของระบบ
# - รับไฟล์อัปโหลด ตรวจสอบชนิด/ขนาดไฟล์ แล้วส่งงานไปยังโมดูลประมวลผลเสียงแต่ละตัว
# - จัดการเรื่อง response, error handling และ cleanup ไฟล์ชั่วคราวหลังประมวลผล

import os
import asyncio
import shutil
import threading
import zipfile
import logging
from uuid import uuid4
from typing import Tuple

# ชุด import ของ FastAPI สำหรับสร้างแอป, รับไฟล์, และส่ง response กลับ
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# import ฟังก์ชันประมวลผลจากโมดูลย่อยของ backend
from backend.process_audio import separate_audio, analyze_audio, pitch_shift_audio
from backend.eq_compressor import apply_compression
from backend.auto_eq_inference import (
    apply_auto_eq_file,
    AutoEQModelLoadError,
    DELTA_CLAMP_DB,
    MIN_DELTA_CLAMP_DB,
    MAX_DELTA_CLAMP_DB,
)

# ไฟล์นี้เป็นจุดรวมการตั้งค่า FastAPI และประกาศ endpoint หลักของระบบทั้งหมด


# การตั้งค่าแอปพลิเคชัน
app = FastAPI()
# logger ใช้บันทึก error หรือสถานะสำคัญของ backend ระหว่างรันจริง
logger = logging.getLogger(__name__)

# โฟลเดอร์ทำงานและข้อจำกัดของไฟล์
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ค่าคอนฟิกระหว่างรัน
# อ่าน origin ที่อนุญาตจาก environment เพื่อให้เปลี่ยนได้โดยไม่ต้องแก้โค้ด
allow_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
# แยกค่า origin ที่คั่นด้วย comma ให้เป็น list สำหรับ CORS middleware
allow_origins = [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]
# เวลาหลังประมวลผลเสร็จที่ไฟล์ชั่วคราวจะถูกลบทิ้งอัตโนมัติ
cleanup_ttl = int(os.getenv("SEPARATE_TTL_SECONDS", "1200"))  # 20 นาทีลบ
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
async def save_upload(file: UploadFile, upload_dir: str = UPLOAD_DIR) -> Tuple[str, str]:
    # ฟังก์ชันช่วยกลางสำหรับ endpoint ที่ต้องรับไฟล์เสียงเข้ามา
    # งานของมันคือ validate ไฟล์, สร้างชื่อไม่ซ้ำ, และเขียนไฟล์ลงดิสก์
    # ดึงชื่อไฟล์เดิมจาก request; ถ้าไม่มีชื่อไฟล์ให้ใช้สตริงว่างแทน
    filename = file.filename or ""

    # แยกนามสกุลไฟล์ออกมาเพื่อตรวจชนิดไฟล์
    _, ext = os.path.splitext(filename)

    # backend นี้รับเฉพาะไฟล์ .wav เท่านั้น ถ้าไม่ใช่ให้ตอบกลับ 400 ทันที
    if ext.lower() != ".wav":
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ WAV (.wav)")

    # อ่านข้อมูลไฟล์ทั้งหมดจาก UploadFile มาเป็น bytes
    data = await file.read()

    # ป้องกันการอัปโหลดไฟล์ใหญ่เกินที่ระบบกำหนด
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    # สร้าง id ไม่ซ้ำเพื่อกันชื่อไฟล์ชนกันและใช้อ้างอิงไฟล์นี้ในขั้นตอนถัดไป
    file_id = str(uuid4())

    # รวม file_id กับชื่อไฟล์เดิมเพื่อให้ได้ชื่อไฟล์ที่เก็บจริงบนดิสก์
    stored_name = f"{file_id}_{filename}"

    # สร้าง path เต็มของไฟล์ที่จะถูกบันทึกลงในโฟลเดอร์ uploads
    input_path = os.path.join(upload_dir, stored_name)

    # เขียนข้อมูลแบบ binary ลงดิสก์
    # เขียนไฟล์แบบ binary ลงดิสก์เพื่อให้โมดูลประมวลผลอื่นอ่านต่อได้
    with open(input_path, "wb") as f:
        f.write(data)

    # คืนทั้ง file_id และ path ของไฟล์ที่บันทึกแล้วให้ endpoint ที่เรียกใช้เอาไปทำงานต่อ
    return file_id, input_path


def schedule_cleanup(path: str, delay: int = 0):
    # ลบไฟล์หรือโฟลเดอร์แบบหน่วงเวลาใน background เพื่อไม่ให้ request หลักต้องรอ


    def _cleanup():
        try:
            # ถ้าเป็นโฟลเดอร์ให้ลบทั้งโฟลเดอร์ ถ้าเป็นไฟล์ให้ลบเฉพาะไฟล์นั้น
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            # cleanup ไม่ควรทำให้เซิร์ฟเวอร์ล่ม จึงแค่ log ปัญหาแล้วจบ
            print(f"cleanup failed for {path}: {exc}")

    # ใช้ Timer แยก background thread เพื่อรอแล้วค่อยลบทีหลัง
    timer = threading.Timer(delay, _cleanup)
    timer.daemon = True
    timer.start()


#เส้น api แยกเสียง
@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    try:
        # บันทึกไฟล์ที่อัปโหลดก่อน แล้วค่อยส่ง path ให้โมดูลแยก stem ทำงานต่อ
        file_id, input_path = await save_upload(file)

        # โฟลเดอร์ผลลัพธ์จะแยกตาม file_id เพื่อไม่ให้แต่ละงานชนกัน
        output_dir = os.path.join("separated", file_id)
        os.makedirs(output_dir, exist_ok=True)
        # รันงานแยก stem ใน thread แยก เพื่อไม่ block event loop ของ FastAPI
        await asyncio.to_thread(separate_audio, input_path, output_dir)

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
        # เก็บ zip และโฟลเดอร์ผลลัพธ์ไว้ชั่วคราวให้ผู้ใช้โหลดได้ก่อน แล้วค่อยลบตาม TTL
        schedule_cleanup(zip_path, cleanup_ttl)
        schedule_cleanup(output_dir, cleanup_ttl)

        # ส่ง URL ที่ frontend ใช้เรียกดาวน์โหลด zip ภายหลัง
        return JSONResponse(
            content={
                "status": "success",
                "file_id": file_id,
                "zip_url": f"http://localhost:8000/download/{file_id}",
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
    zip_filename = f"{file_id}_separated.zip"
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


@app.get("/separated/{file_id}/{stem}.wav")
async def get_stem(file_id: str, stem: str):
    # ส่งคืน stem เดี่ยวให้ตัวเล่นหลายแทร็กเรียกไปเปิดทีละไฟล์
    filename = f"{stem}.wav"
    folder = os.path.join("separated", file_id)
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        # frontend ใช้ endpoint นี้โหลด stem เฉพาะตัวไปทำ multitrack playback
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"status": "error", "message": f"ไม่พบไฟล์ {stem}.wav"})


# เส้นทาง Auto-EQ แบบ AI
@app.post("/apply-eq-ai")
async def apply_eq_ai(
    file: UploadFile = File(...),
    delta_clamp_db: float = Query(
        DELTA_CLAMP_DB,
        ge=MIN_DELTA_CLAMP_DB,
        le=MAX_DELTA_CLAMP_DB,
        description="Auto-EQ delta clamp in dB",
    ),
    genre: str = Query("pop", description="แนวเพลง เช่น pop, rock, trap, country, soul"),
    delta_clamp_db: float = Query(
        DELTA_CLAMP_DB,
        ge=MIN_DELTA_CLAMP_DB,
        le=MAX_DELTA_CLAMP_DB,
        description="เพดานการปรับ EQ ต่อจุดในหน่วย dB",
    ),
):
    try:
        # บันทึกไฟล์ก่อน แล้วส่ง path ไปให้โมเดล Auto-EQ ประมวลผล
        file_id, input_path = await save_upload(file)
        # ตั้งชื่อไฟล์ผลลัพธ์ให้สะท้อนว่าเป็นงาน EQ AI และใช้ genre อะไร
        output_filename = f"{file_id}_eq_ai_{genre}.wav"
        output_path = os.path.join("eq_applied", output_filename)
        # inference รันใน worker thread เพื่อไม่ block FastAPI main loop
        result_path = await asyncio.to_thread(
            apply_auto_eq_file,
            input_path,
            output_path,
            genre,
            delta_clamp_db,
        )
        return FileResponse(
            result_path,
            media_type="audio/wav",
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
):
    try:
        # งาน compressor ใช้ทั้ง preset และค่าที่ผู้ใช้ override ผ่าน query string
        _, input_path = await save_upload(file)
        # เรียกใช้ฟังก์ชัน DSP ใน thread แยก เพราะเป็นงานคำนวณและอ่านเขียนไฟล์
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
        return FileResponse(
            output_path,
            media_type="audio/wav",
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
async def pitch_shift(file: UploadFile = File(...), steps: float = 0):
    try:
        # บันทึกไฟล์ก่อน จากนั้นสร้างไฟล์ผลลัพธ์ที่ถูก shift pitch แล้วส่งกลับ
        file_id, input_path = await save_upload(file)
        # ตั้งชื่อไฟล์ผลลัพธ์ใหม่เพื่อไม่เขียนทับต้นฉบับ
        output_filename = f"{file_id}_pitch.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        # ฟังก์ชัน pitch_shift_audio จะคืน path ของไฟล์ผลลัพธ์กลับมา
        # ใช้ to_thread เช่นเดียวกับ endpoint อื่นที่ทำงานประมวลผลเสียงหนัก ๆ
        result_path = await asyncio.to_thread(pitch_shift_audio, input_path, steps, output_path)
        return FileResponse(
            result_path,
            media_type="audio/wav",
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
async def analyze(file: UploadFile = File(...)):
    try:
        # endpoint นี้คืนข้อมูลวิเคราะห์เป็น JSON ไม่ได้สร้างไฟล์เสียงใหม่
        _, input_path = await save_upload(file)
        # วิเคราะห์เสียงใน thread แยก แล้วนำผลลัพธ์ JSON กลับมาโดยตรง
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
