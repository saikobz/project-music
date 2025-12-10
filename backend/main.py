import os
import asyncio
import shutil
import threading
import zipfile
from uuid import uuid4
from typing import Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.process_audio import separate_audio, analyze_audio, pitch_shift_audio
from backend.eq_compressor import apply_eq_to_file, apply_compression
from backend.auto_eq_inference import apply_auto_eq_file


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

allow_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allow_origins = [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]
cleanup_ttl = int(os.getenv("SEPARATE_TTL_SECONDS", "21600"))  # 6 hours by default
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


async def save_upload(file: UploadFile, upload_dir: str = UPLOAD_DIR) -> Tuple[str, str]:
    """Validate and save upload. Returns (file_id, path)."""
    filename = file.filename or ""
    _, ext = os.path.splitext(filename)
    if ext.lower() != ".wav":
        raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ WAV (.wav)")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="ไฟล์ต้องมีขนาดไม่เกิน 100MB")

    file_id = str(uuid4())
    stored_name = f"{file_id}_{filename}"
    input_path = os.path.join(upload_dir, stored_name)
    with open(input_path, "wb") as f:
        f.write(data)
    return file_id, input_path


def schedule_cleanup(path: str, delay: int = 0):
    """Schedule cleanup to prevent disk bloat."""

    def _cleanup():
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            print(f"cleanup failed for {path}: {exc}")

    timer = threading.Timer(delay, _cleanup)
    timer.daemon = True
    timer.start()


@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    try:
        file_id, input_path = await save_upload(file)

        output_dir = os.path.join("separated", file_id)
        os.makedirs(output_dir, exist_ok=True)
        await asyncio.to_thread(separate_audio, input_path, output_dir)

        zip_filename = f"{file_id}_separated.zip"
        zip_path = os.path.join(UPLOAD_DIR, zip_filename)
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for name in files:
                    file_path = os.path.join(root, name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        if os.path.exists(input_path):
            os.remove(input_path)
        schedule_cleanup(zip_path, cleanup_ttl)
        schedule_cleanup(output_dir, cleanup_ttl)

        return JSONResponse(
            content={
                "status": "success",
                "file_id": file_id,
                "zip_url": f"http://localhost:8000/download/{file_id}",
            }
        )

    except HTTPException as http_exc:
        raise http_exc
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
            filename="separated.zip",
        )
    return JSONResponse(status_code=404, content={"status": "error", "message": "ไม่พบไฟล์ zip สำหรับดาวน์โหลด"})


@app.get("/separated/{file_id}/{stem}.wav")
async def get_stem(file_id: str, stem: str):
    filename = f"{stem}.wav"
    folder = os.path.join("separated", file_id)
    path = os.path.join(folder, filename)

    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"status": "error", "message": f"ไม่พบไฟล์ {stem}.wav"})


@app.post("/apply-eq")
async def apply_eq(
    file: UploadFile = File(...),
    genre: str = Query("pop", description="แนวเพลง เช่น pop, rock, trap, country, soul"),
):
    try:
        _, input_path = await save_upload(file)
        output_path = await asyncio.to_thread(apply_eq_to_file, input_path, genre)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path),
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/apply-eq-ai")
async def apply_eq_ai(
    file: UploadFile = File(...),
    genre: str = Query("pop", description="แนวเพลง เช่น pop, rock, trap, country, soul"),
):
    try:
        file_id, input_path = await save_upload(file)
        output_filename = f"{file_id}_eq_ai_{genre}.wav"
        output_path = os.path.join("eq_applied", output_filename)
        result_path = await asyncio.to_thread(apply_auto_eq_file, input_path, output_path)
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
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/apply-compressor")
async def apply_compressor(
    file: UploadFile = File(...),
    strength: str = "medium",
    genre: str = Query("general", description="แนวเพลง เช่น pop, rock, trap, country, soul"),
):
    try:
        _, input_path = await save_upload(file)
        output_path = await asyncio.to_thread(apply_compression, input_path, strength, genre)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=os.path.basename(output_path),
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/pitch-shift")
async def pitch_shift(file: UploadFile = File(...), steps: float = 0):
    try:
        file_id, input_path = await save_upload(file)
        output_filename = f"{file_id}_pitch.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        # pitch_shift_audio will return the output path
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
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        _, input_path = await save_upload(file)
        result = await asyncio.to_thread(analyze_audio, input_path)
        return JSONResponse(content=result)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)
