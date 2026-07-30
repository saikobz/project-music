# backend/config.py
# ศูนย์กลางการตั้งค่า ตัวแปรระบบ ค่าคงที่ และสโคปไดเรกทอรีสำหรับ Backend ทั้งหมด

import os

# โฟลเดอร์สำหรับเก็บไฟล์ชั่วคราวแต่ละขั้นตอน
DIR_UPLOADS = "uploads"
DIR_SEPARATED = "separated"
DIR_EQ_APPLIED = "eq_applied"
DIR_COMPRESSED = "compressed"

# รายการโฟลเดอร์ทั้งหมดที่ต้องกวาดลบไฟล์หมดอายุ (รวม compressed ไว้แล้ว)
ALL_CLEANUP_DIRS = [DIR_UPLOADS, DIR_SEPARATED, DIR_EQ_APPLIED, DIR_COMPRESSED]

# ขนาดไฟล์อัปโหลดสูงสุด (100MB)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# ระยะเวลาหมดอายุและช่วงเวลารันระบบ cleanup (วินาที)
DEFAULT_CLEANUP_TTL_SECONDS = int(os.getenv("SEPARATE_TTL_SECONDS", "1200"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))

# ข้อจำกัด Concurrency
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))

# สเต็มเป้าหมายและ Genres ที่รองรับในระบบ
STEM_TARGETS = ("vocals", "drums", "bass", "other")
SUPPORTED_GENRES = ("pop", "rock", "trap", "country", "soul")
