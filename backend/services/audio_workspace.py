# backend/services/audio_workspace.py
# Deep Module: ศูนย์กลางจัดการ Audio Job Storage, File Artifacts, Format Conversions และ TTL Cleanup

import os
import shutil
import zipfile
import asyncio
import logging
import soundfile as sf
import numpy as np
from fastapi import UploadFile, HTTPException
from backend.services.storage import save_upload, convert_to_mp3, UPLOAD_DIR, processing_semaphore
from backend.services.job_manager import job_manager
from backend.config import ALL_CLEANUP_DIRS, DEFAULT_CLEANUP_TTL_SECONDS, DIR_SEPARATED

logger = logging.getLogger(__name__)


class AudioJobWorkspace:
    """ Deep Module สำหรับจัดการวงจรชีวิตไฟล์เสียงและโฟลเดอร์ผลลัพธ์ทั้งหมด """

    def __init__(self, upload_dir: str = UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    async def save_upload_file(
        self,
        file: UploadFile,
        trim_start: float | None = None,
        trim_end: float | None = None
    ) -> tuple[str, str]:
        """รับไฟล์อัปโหลด บันทึกลงดิสก์ และจัดการตัดช่วงเวลาเสียง"""
        return await save_upload(file, self.upload_dir, trim_start, trim_end)

    def prepare_output_dir(self, file_id: str, category: str = "separated") -> str:
        """เตรียมโฟลเดอร์สำหรับผลลัพธ์งานประมวลผลพร้อมลงทะเบียน Job"""
        safe_file_id = os.path.basename(file_id)
        out_dir = os.path.join(category, safe_file_id)
        os.makedirs(out_dir, exist_ok=True)
        job_manager.register_job(safe_file_id, out_dir)
        return out_dir

    def mark_job_completed(self, file_id: str):
        """ระบุว่า Job ประมวลผลเสร็จสิ้น"""
        job_manager.complete_job(os.path.basename(file_id))

    async def bundle_separated_stems_zip(
        self,
        file_id: str,
        output_dir: str,
        export_format: str = "wav"
    ) -> str:
        """แปลงไฟล์และบีบอัด Stem ทั้งหมดลงใน Zip Archive"""
        self.mark_job_completed(file_id)

        if export_format == "mp3":
            for root, _, files in os.walk(output_dir):
                for name in files:
                    if name.lower().endswith(".wav"):
                        wav_path = os.path.join(root, name)
                        await asyncio.to_thread(convert_to_mp3, wav_path)

        safe_file_id = os.path.basename(file_id)
        zip_filename = f"{safe_file_id}_separated.zip"
        zip_path = os.path.join(self.upload_dir, zip_filename)

        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for name in files:
                    file_path = os.path.join(root, name)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

        return zip_path

    def get_zip_path(self, file_id: str) -> str | None:
        """ค้นหาไฟล์ zip สำหรับดาวน์โหลด"""
        safe_file_id = os.path.basename(file_id)
        zip_filename = f"{safe_file_id}_separated.zip"
        zip_path = os.path.join(self.upload_dir, zip_filename)
        return zip_path if os.path.exists(zip_path) else None

    def get_separated_stem_path(self, file_id: str, filename: str) -> str | None:
        """ค้นหาไฟล์ Stem เดี่ยวในโฟลเดอร์ผลลัพธ์"""
        safe_file_id = os.path.basename(file_id)
        safe_filename = os.path.basename(filename)
        out_dir = job_manager.get_job_directory(safe_file_id) or os.path.join(DIR_SEPARATED, safe_file_id)
        target_path = os.path.join(out_dir, safe_filename)
        return target_path if os.path.exists(target_path) else None

    def cleanup_expired_files(self, ttl_seconds: int = DEFAULT_CLEANUP_TTL_SECONDS):
        """กวาดลบไฟล์ชั่วคราวและโฟลเดอร์ที่หมดอายุในทุกหมวดหมู่"""
        import time
        now = time.time()
        directories = ALL_CLEANUP_DIRS

        for category_dir in directories:
            if not os.path.exists(category_dir):
                continue
            for item in os.listdir(category_dir):
                item_path = os.path.join(category_dir, item)
                try:
                    mtime = os.path.getmtime(item_path)
                    if now - mtime > ttl_seconds:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            logger.info(f"Cleaned up expired file: {item_path}")
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                            logger.info(f"Cleaned up expired directory: {item_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {item_path}: {e}")


# Singleton instance
audio_workspace = AudioJobWorkspace()
