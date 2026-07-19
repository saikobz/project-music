import os
import time
import asyncio
import shutil
import logging

logger = logging.getLogger(__name__)

# โฟลเดอร์ที่ต้องตรวจสอบเพื่อลบไฟล์เก่า
CLEANUP_DIRS = ["uploads", "separated", "eq_applied"]

async def periodic_cleanup(interval_seconds: int = 300, ttl_seconds: int = 1200):
    """
    งานเบื้องหลังสำหรับกวาดลบไฟล์/โฟลเดอร์ที่เก่าเกินเวลาที่กำหนด (ttl_seconds)
    จะรันวนลูปทุกๆ interval_seconds
    """
    logger.info(f"เริ่มต้นระบบ Cleanup กวาดขยะทุก {interval_seconds} วินาที (ลบไฟล์เก่ากว่า {ttl_seconds} วิ)")
    while True:
        try:
            now = time.time()
            for d in CLEANUP_DIRS:
                if not os.path.exists(d):
                    continue
                
                # อ่านรายการไฟล์และโฟลเดอร์ทั้งหมดในไดเรกทอรี
                for filename in os.listdir(d):
                    file_path = os.path.join(d, filename)
                    
                    try:
                        # เช็คเวลาสร้าง/แก้ไขล่าสุด
                        mtime = os.path.getmtime(file_path)
                        
                        if now - mtime > ttl_seconds:
                            # เก่าเกินเวลา ลบทิ้ง
                            if os.path.isdir(file_path):
                                shutil.rmtree(file_path, ignore_errors=True)
                                logger.info(f"ลบโฟลเดอร์หมดอายุ: {file_path}")
                            else:
                                os.remove(file_path)
                                logger.info(f"ลบไฟล์หมดอายุ: {file_path}")
                    except Exception as e:
                        logger.error(f"ไม่สามารถลบ {file_path} ได้: {e}")
                        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในลูป Cleanup: {e}")
            
        # รอจนกว่าจะถึงรอบถัดไป
        await asyncio.sleep(interval_seconds)
