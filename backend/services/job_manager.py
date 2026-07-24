# backend/services/job_manager.py
# ระบบจัดการวงจรชีวิตงานประมวลผลเสียง (Job Session Lifecycle)

import os
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AudioJobSessionManager:
    """จัดการสถานะ Job ID, โฟลเดอร์ผลลัพธ์ และเวลาหมดอายุเพื่อการCleanup อย่างปลอดภัย"""

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def register_job(self, file_id: str, output_dir: str) -> None:
        self._jobs[file_id] = {
            "output_dir": output_dir,
            "created_at": time.time(),
            "status": "processing",
        }

    def complete_job(self, file_id: str) -> None:
        if file_id in self._jobs:
            self._jobs[file_id]["status"] = "completed"

    def get_job_directory(self, file_id: str) -> Optional[str]:
        job = self._jobs.get(file_id)
        if job:
            return job["output_dir"]
        folder = os.path.join("separated", file_id)
        if os.path.exists(folder):
            return folder
        return None


# Global instance สำหรับเรียกใช้ทั่วทั้งแอป
job_manager = AudioJobSessionManager()
