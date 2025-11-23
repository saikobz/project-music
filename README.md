# HarmoniQ - การตั้งค่า รัน และเช็กสุขภาพ

โครงสร้าง:
- Backend: `backend/` (FastAPI + PyTorch/Open-Unmix)
- Frontend: Next.js ที่รากโปรเจกต์ (โฟลเดอร์ `app/`)

## สิ่งที่ต้องมี
- Python 3.10 (64-bit) พร้อม `venv`
- Node.js 20+ และ npm
- FFmpeg อยู่ใน PATH
- อินเทอร์เน็ตครั้งแรกเพื่อดาวน์โหลด Torch/Open-Unmix (ขนาดใหญ่)
- GPU ไม่บังคับ แต่มี CUDA จะเร่งการแยกสเต็ม

## Backend (FastAPI)
```
cd backend
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
- ไฟล์อัปโหลดเก็บที่ `uploads/`; ไฟล์สเต็มที่แยกเก็บที่ `separated/` (ถูก ignore)
- ตั้งค่า CORS ผ่าน env `ALLOWED_ORIGINS` (คั่นด้วย comma) ไม่ตั้งค่าจะใช้ `http://localhost:3000`

## Frontend (Next.js 15)
```
cd C:\Users\nopma\Desktop\project-music
npm install
npm run dev   # เริ่มที่พอร์ต 3000
```
- UI เรียก backend ที่ `http://localhost:8000/...`

## การทดสอบ/Health Check
- เช็ก backend ว่ารันอยู่หรือไม่: `curl http://localhost:8000/health` (คาดหวัง `{"status":"ok"}`)
- ใช้ Postman/เบราว์เซอร์เปิด `/health` ได้เหมือนกัน

## ทิปส์
- ถ้าเปลี่ยนพอร์ต/โดเมน frontend ให้ตั้ง `ALLOWED_ORIGINS` ให้ตรงก่อนรัน backend
- ตรวจ FFmpeg หลังติดตั้งด้วย `ffmpeg -version`
- ดีเพนเดนซี Torch/Open-Unmix ใหญ่ ใช้เวลาติดตั้งและพื้นที่ดิสก์พอสมควร
