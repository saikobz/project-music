import os
import sys
import json
import urllib.request
from urllib.error import URLError, HTTPError
import time
import socket

# กำหนด Timeout สูงสุดของการเชื่อมต่อ (ระดับ Socket) 
# เพื่อป้องกันอาการค้าง (Hang) ระหว่างที่กำลังอ่านข้อมูลไฟล์แล้วเน็ตหลุด
socket.setdefaulttimeout(15.0)

MODEL_DIR = os.path.join("backend", "models", "umxl")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = ["vocals", "drums", "bass", "other"]

# URL สำหรับดาวน์โหลดโมเดล umxl จาก Zenodo
# ข้อมูลเหล่านี้อ้างอิงตรงจากโมเดลมาตรฐานในแพ็คเกจ openunmix
TARGET_URLS = {
    "bass": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/bass-2ca1ce51.pth",
    "drums": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/drums-69e0ebd4.pth",
    "other": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/other-c8c5b3e6.pth",
    "vocals": "https://zenodo.org/api/files/f8209c3e-ba60-48cf-8e79-71ae65beca61/vocals-bccbd9aa.pth",
}

def download_file(url, filepath):
    """
    ดาวน์โหลดไฟล์พร้อมระบบ Resume (ดาวน์โหลดต่อจากจุดเดิมเมื่อเน็ตหลุด)
    """
    temp_filepath = filepath + ".tmp"
    
    # หาตำแหน่งบิตเริ่มต้น (กรณีดาวน์โหลดค้างไว้จากรอบก่อน)
    initial_bytes = 0
    if os.path.exists(temp_filepath):
        initial_bytes = os.path.getsize(temp_filepath)
        
    headers = {}
    if initial_bytes > 0:
        # ใช้ HTTP Range Request เพื่อดาวน์โหลดต่อจากเดิม
        headers['Range'] = f"bytes={initial_bytes}-"
        
    req = urllib.request.Request(url, headers=headers)
    
    try:
        # ตั้ง Timeout ไว้ที่ 20 วินาที เพื่อไม่ให้ค้างยาวเมื่อเน็ตมีปัญหา
        with urllib.request.urlopen(req, timeout=20) as response:
            content_length = int(response.headers.get('Content-Length', 0))
            
            # ตรวจสอบว่าฝั่ง Server อนุญาตให้ดาวน์โหลดต่อ (Status 206) หรือส่งไฟล์มาใหม่หมด (Status 200)
            if response.status == 206:
                total_size = content_length + initial_bytes
                mode = 'ab' # เขียนต่อท้ายไฟล์เดิม (Append Binary)
                print(f"  ดาวน์โหลดต่อจากจุดเดิม (Resume) ที่ตำแหน่ง: {initial_bytes / (1024*1024):.2f} MB")
            else:
                total_size = content_length
                initial_bytes = 0
                mode = 'wb' # เขียนทับไฟล์ใหม่ (Write Binary)
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            
            downloaded = initial_bytes
            start_time = time.time()
            
            with open(temp_filepath, mode) as f:
                while True:
                    # อ่านและเขียนครั้งละ 64KB
                    chunk = response.read(1024 * 64)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # คำนวณความเร็วและเปอร์เซ็นต์
                    percent = (downloaded / total_size) * 100 if total_size else 0
                    elapsed = time.time() - start_time
                    speed = (downloaded - initial_bytes) / elapsed / 1024 if elapsed > 0 else 0
                    
                    sys.stdout.write(
                        f"\r  Progress: {percent:6.2f}% | "
                        f"{downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB | "
                        f"Speed: {speed:6.1f} KB/s"
                    )
                    sys.stdout.flush()
            print()
            
            # เมื่อดาวน์โหลดเรียบร้อย ให้เปลี่ยนชื่อเป็นไฟล์ .pth จริง
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_filepath, filepath)
            return True
            
    except HTTPError as e:
        if e.code == 416: # Range Not Satisfiable (อาจเกิดจากขนาดไฟล์ฝั่งเซิร์ฟเวอร์เท่ากับไฟล์ในเครื่องแล้ว)
            # ลบและเริ่มโหลดใหม่ทั้งหมดเพื่อความปลอดภัย
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return download_file(url, filepath)
        print(f"\n  HTTP Error {e.code}: {e.reason}")
        return False
    except URLError as e:
        print(f"\n  Network Connection Error: {e.reason}")
        return False
    except (socket.timeout, TimeoutError):
        print(f"\n  Connection Timed Out: การเชื่อมต่อหยุดนิ่ง ค้าง หรือขาดหาย")
        return False
    except Exception as e:
        print(f"\n  Error: {e}")
        return False

def create_config_files():
    """
    สร้างไฟล์คอนฟิก JSON สำหรับ Open-Unmix (umxl) ให้สามารถโหลดแบบ Local ได้โดยไม่ต้องอาศัยอินเทอร์เน็ต
    """
    print("\nสร้างไฟล์ตั้งค่าความถี่สำหรับการทำงานแบบออฟไลน์ (Local Configs)...")
    
    # 1. สร้าง separator.json
    separator_config = {
        "sample_rate": 44100.0,
        "nfft": 4096,
        "nhop": 1024,
        "nb_channels": 2
    }
    with open(os.path.join(MODEL_DIR, "separator.json"), "w") as f:
        json.dump(separator_config, f, indent=2)
        
    # 2. สร้างไฟล์คอนฟิกสำหรับแต่ละ Target (vocals, drums, bass, other)
    target_config = {
        "args": {
            "nfft": 4096,
            "nb_channels": 2,
            "hidden_size": 1024
        }
    }
    for target in TARGETS:
        with open(os.path.join(MODEL_DIR, f"{target}.json"), "w") as f:
            json.dump(target_config, f, indent=2)
            
    print("  [สำเร็จ] สร้างสเปค JSON สำหรับใช้งานออฟไลน์เสร็จสมบูรณ์!")

def main():
    print("==========================================================")
    print("AI Music Stem Separator - Robust Model Downloader")
    print("==========================================================")
    print(f"ไฟล์ทั้งหมดจะถูกบันทึกลงใน: {MODEL_DIR}")
    print("สคริปต์นี้มีระบบช่วยดาวน์โหลดต่อจากเดิม (Resume) หากเครือข่ายมีปัญหา")
    print("==========================================================")
    
    success = True
    for target in TARGETS:
        filename = f"{target}.pth"
        filepath = os.path.join(MODEL_DIR, filename)
        url = TARGET_URLS[target]
        
        # หากไฟล์สมบูรณ์แล้ว ให้ข้ามการดาวน์โหลด
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 108000000:
            print(f"-> โมเดล {target.upper()} มีความพร้อมใช้งานอยู่แล้ว (ข้ามการดาวน์โหลด)")
            continue
            
        print(f"\n-> กำลังเตรียมดาวน์โหลดโมเดล {target.upper()}...")
        
        # ลองดาวน์โหลดซ้ำสูงสุด 3 ครั้งในสคริปต์เมื่อล้มเหลว
        download_ok = False
        for attempt in range(1, 4):
            if attempt > 1:
                print(f"  [Retry] พยายามดาวน์โหลดอีกครั้ง ครั้งที่ {attempt}/3...")
                time.sleep(2)
            if download_file(url, filepath):
                download_ok = True
                break
                
        if not download_ok:
            print(f"  [ล้มเหลว] ไม่สามารถดาวน์โหลดโมเดล {target.upper()} ได้สำเร็จ")
            success = False
            break
            
    if success:
        # เมื่อโหลด .pth ครบทั้งหมดแล้ว ให้สร้างไฟล์คอนฟิกสเปค JSON
        create_config_files()
        print("\n==========================================================")
        print("[ยินดีด้วย] ระบบดาวน์โหลดโมเดล Open-Unmix ครบถ้วนแล้ว!")
        print("คุณสามารถทดลองใช้งานฟังก์ชันแยกสเตมได้ทันทีโดยไม่ต้องเชื่อมต่อเน็ต")
        print("==========================================================")
    else:
        print("\n==========================================================")
        print("[คำเตือน] ดาวน์โหลดล้มเหลวชั่วคราวเนื่องจากปัญหาเครือข่าย")
        print("โปรดเช็คความเสถียรของอินเทอร์เน็ตแล้วลองรันสคริปต์นี้อีกครั้ง")
        print("==========================================================")

if __name__ == "__main__":
    main()
