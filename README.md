# HarmoniQ - การตั้งค่าและวิธีรัน

โครงสร้างโปรเจกต์ :
- Backend: `backend/` (FastAPI + PyTorch/Open-Unmix)
- Frontend: Next.js ในรากโปรเจกต์ (ใช้โฟลเดอร์ `app/`)

## สิ่งที่ต้องมี (Prerequisites)
- Python 3.10 (64-bit) พร้อม `venv`
- Node.js 20+ และ npm
- FFmpeg อยู่ใน PATH (pydub/librosa ใช้สำหรับอ่าน/เขียนเสียง)
- ใข้อินเทอร์เน็ตครั้งแรกเพื่อดาวน์โหลด Torch/Open-Unmix ซึ่งมีขนาดใหญ่
- GPU ไม่บังคับ แต่มี CUDA จะช่วยให้การแยกสเต็มเร็วขึ้น

## Backend (FastAPI)
```
cd backend
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
- ไฟล์อัปโหลดจะเก็บใน `uploads/`; ไฟล์สเต็มที่แยกแล้วอยู่ใน `separated/` (ถูก ignore ใน git)
- CORS เปิดให้ `http://localhost:3000` เท่านั้น หากเสิร์ฟ frontend ที่โดเมน/พอร์ตอื่น แก้ที่ตัวแปร `allow_origins` ใน `backend/main.py`

## Frontend (Next.js 15)
```
cd C:\Users\nopma\Desktop\project-music
npm install
npm run dev   # ค่าเริ่มต้นพอร์ต 3000
```
- UI จะเรียก backend ที่ `http://localhost:8000/...`

## 
- ถ้าปรับพอร์ตหรือโดเมนของ frontend อย่าลืมอัปเดต CORS ใน backend ให้ตรง
- ตรวจสอบ FFmpeg ด้วยคำสั่ง `ffmpeg -version` หลังติดตั้ง
- ดีเพนเดนซี Torch/Open-Unmix มีขนาดใหญ่ ต้องใช้เวลาติดตั้งและพื้นที่ดิสก์พอสมควร
