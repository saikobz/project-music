# backend/main.py
# - ศูนย์กลางของ FastAPI backend และจุดลงทะเบียน API Routers
# - กำหนดค่า CORS, Lifespan tasks และ Health Check

import os
import asyncio
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from backend.cleanup_task import periodic_cleanup
from backend.routers import stems, audio_ops

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# เวลาหลังประมวลผลเสร็จที่ไฟล์ชั่วคราวจะถูกลบทิ้งอัตโนมัติ (Default: 20 นาที)
cleanup_ttl = int(os.getenv("SEPARATE_TTL_SECONDS", "1200"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # เริ่มต้น background task กวาดลบไฟล์ชั่วคราวที่หมดอายุ
    cleanup_task = asyncio.create_task(periodic_cleanup(interval_seconds=300, ttl_seconds=cleanup_ttl))
    logger.info("FastAPI backend started with background cleanup task.")
    yield
    # เมื่อปิดการทำงานเซิร์ฟเวอร์
    cleanup_task.cancel()
    logger.info("FastAPI backend shutting down.")


app = FastAPI(
    title="HarmoniQ API Backend",
    description="ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI",
    version="1.0.0",
    lifespan=lifespan,
)

# อ่าน Origin ที่อนุญาตจาก environment
allow_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
allow_origins = [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]

# เปิดใช้งาน CORS สำหรับ Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ลงทะเบียน Routers
app.include_router(stems.router)
app.include_router(audio_ops.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """C6: จัดการ exception ที่ไม่คาดคิดทั้งหมดไว้ที่จุดเดียว (เดิมซ้ำทุก router)

    - log รายละเอียดฝั่ง server เท่านั้น
    - คืนข้อความ generic (M17: กันรั่ว error detail ภายในออกไปที่ client)
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์"},
    )


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint สำหรับตรวจสอบสถานะเซิร์ฟเวอร์"""
    return {"status": "ok", "service": "HarmoniQ API Backend"}
