from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import os
from uuid import uuid4
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import zipfile
import shutil

from backend.process_audio import separate_audio, analyze_audio, pitch_shift_audio
from backend.eq_compressor import apply_eq_to_file
from backend.eq_compressor import apply_compression


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # หรือ ["*"] ถ้าทดสอบ local
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        # 1. แยกเสียงไปไว้ที่ separated/{file_id}/
        output_dir = os.path.join("separated", file_id)
        os.makedirs(output_dir, exist_ok=True)
        await asyncio.to_thread(separate_audio, input_path, output_dir)  # ✅ ใช้ path แยกตาม file_id

        # 2. สร้าง zip
        zip_filename = f"{file_id}_separated.zip"
        zip_path = os.path.join(UPLOAD_DIR, zip_filename)

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for name in files:
                    file_path = os.path.join(root, name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        return JSONResponse(content={
            "status": "success",
            "file_id": file_id,
            "zip_url": f"http://localhost:8000/download/{file_id}"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/download/{file_id}")
async def download_zip(file_id: str):
    zip_filename = f"{file_id}_separated.zip"
    zip_path = os.path.join(UPLOAD_DIR, zip_filename)

    if os.path.exists(zip_path):
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="separated.zip"
        )
    else:
        return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ zip"})


@app.get("/separated/{file_id}/{stem}.wav")
async def get_stem(file_id: str, stem: str):
    filename = f"{stem}.wav"
    folder = os.path.join("separated", file_id)
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    else:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"ไม่พบไฟล์ {stem}.wav"})


@app.post("/apply-eq")
async def apply_eq(file: UploadFile = File(...), target: str = "vocals"):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        output_path = await asyncio.to_thread(apply_eq_to_file, input_path, target) 
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/apply-compressor")
async def apply_compressor(file: UploadFile = File(...), strength: str = "medium"):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        output_path = await asyncio.to_thread(apply_compression, input_path, strength)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path)
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/pitch-shift")
async def pitch_shift(file: UploadFile = File(...), steps: float = 0):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        output_filename = f"{file_id}_pitch.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        result_path = await asyncio.to_thread(pitch_shift_audio, input_path, steps, output_path)
        return FileResponse(
            result_path,
            media_type="audio/wav",
            filename=os.path.basename(result_path)
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

            
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        result = await asyncio.to_thread(analyze_audio, input_path)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    steps = [
        {"message": "📤 กำลังอัปโหลดไฟล์...", "progress": 10},
        {"message": "🎧 กำลังแยกเสียงร้องและดนตรี...", "progress": 40},
        {"message": "🎚️ ปรับ EQ และใส่ Compressor...", "progress": 75},
        {"message": "💾 บันทึกไฟล์และ metadata...", "progress": 100}
    ]

    results = []
    for step in steps:
        await asyncio.sleep(2)
        results.append(step)

    return JSONResponse(content={"steps": results})
