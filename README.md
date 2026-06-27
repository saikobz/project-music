# HarmoniQ – AI Music Stem & Mastering Toolkit

เว็บแอปสำหรับแยกเสียงจากไฟล์ WAV ด้วย AI พร้อม EQ, Compressor, Pitch Shift, และเครื่องมือวิเคราะห์คีย์/เทมโป/โทนเสียง มีเครื่องเล่นหลายสเตมบนหน้าเว็บให้ลองฟังและดาวน์โหลดรายแทร็ก

## คุณสมบัติหลัก
- แยกเสียงด้วย Open-Unmix (vocals, drums, bass, other) แล้วแพ็กเป็น ZIP
- ใส่ EQ แบบ preset ตามสเตม, Compressor หลายระดับ, และ Pitch Shift
- Auto-EQ (AI) ด้วยโมเดล `autoeq_cnn_v1.pt` (Conv-BN-Conv 16ch, ประมวลผลเป็นบล็อก 5 วินาที)
- วิเคราะห์เพลง: Tempo, Key, Median Pitch (โน้ต)
- ตัวเล่นเสียงแบบ Multi-Stem (ควบคุม mute/play/seek ต่อแทร็ก) และตัวเล่นไฟล์เดี่ยวแบบ waveform

## ความต้องการระบบ
- Python 3.10 (64-bit)
- Node.js 20+
- FFmpeg ต้องอยู่ใน `PATH`
- GPU + CUDA (ถ้ามี) จะช่วยให้ Open-Unmix ทำงานเร็วขึ้น แต่ไม่บังคับ

## เริ่มต้นเร็ว (Quick Start)

> ตัวอย่างคำสั่งด้านล่างอ้างอิงการใช้งานบน Windows PowerShell จากโฟลเดอร์รากของโปรเจกต์

1. ติดตั้ง backend dependencies และรัน API

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

2. เปิดเทอร์มินัลอีกอัน แล้วติดตั้ง frontend dependencies

```powershell
npm install
```

3. สร้างไฟล์ `.env.local` สำหรับ frontend

```powershell
"NEXT_PUBLIC_API_BASE=http://localhost:8000" | Set-Content .env.local
```

4. รันหน้าเว็บ

```powershell
npm run dev
```

5. เปิดใช้งาน

- Frontend: `http://localhost:3000`
- Backend health check: `http://localhost:8000/health`

## การติดตั้งแบบละเอียด

### 1) ติดตั้ง Backend (FastAPI)

รันจากโฟลเดอร์รากของโปรเจกต์:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

จากนั้นรัน backend:

```powershell
uvicorn backend.main:app --reload --port 8000
```

สิ่งที่ควรรู้:

- backend รับเฉพาะไฟล์ `.wav`
- ขนาดไฟล์อัปโหลดสูงสุด `100MB` ต่อไฟล์
- โมเดล Auto-EQ อยู่ใน `backend/models/` แล้ว ไม่ต้องดาวน์โหลดเพิ่ม

### 2) ติดตั้ง Frontend (Next.js 15)

รันจากโฟลเดอร์รากของโปรเจกต์:

```powershell
npm install
npm run dev
```

คำสั่งอื่นที่ใช้บ่อย:

```powershell
npm run build
npm start
```

### 3) การตั้งค่า Environment Variables

#### Frontend

frontend จะอ่านค่า `NEXT_PUBLIC_API_BASE` และถ้าไม่ตั้งค่า จะใช้ `http://localhost:8000` โดยอัตโนมัติ

ตัวอย่างไฟล์ `.env.local`:

```env
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

#### Backend

backend ใช้ environment variables หลักดังนี้:

- `ALLOWED_ORIGINS` ค่าเริ่มต้นคือ `http://localhost:3000`
- `SEPARATE_TTL_SECONDS` ค่าเริ่มต้นคือ `1200` วินาที (`20` นาที)

ตัวอย่างการตั้งค่าใน PowerShell ก่อนรัน backend:

```powershell
$env:ALLOWED_ORIGINS = "http://localhost:3000"
$env:SEPARATE_TTL_SECONDS = "1200"
uvicorn backend.main:app --reload --port 8000
```

## วิธีใช้งานหลังติดตั้ง

1. เปิด backend ให้ทำงานที่พอร์ต `8000`
2. เปิด frontend ที่พอร์ต `3000`
3. เข้าเว็บและอัปโหลดไฟล์ `.wav`
4. ใช้งานฟีเจอร์แยกสเตม, EQ, Compressor, Pitch Shift และการวิเคราะห์เพลงจากหน้าเว็บ

เอาต์พุตหลักของระบบจะถูกสร้างในโฟลเดอร์ต่อไปนี้:

- `uploads/` ไฟล์ต้นฉบับและไฟล์ ZIP ที่ดาวน์โหลดกลับ
- `separated/` ไฟล์สเตมที่แยกแล้ว
- `eq_applied/` ไฟล์ที่ผ่าน EQ หรือ Auto-EQ แล้ว
- `compressed/` ไฟล์ที่ผ่าน compression แล้ว

## หมายเหตุ / แก้ปัญหาเบื้องต้น

- ถ้าคำสั่ง `ffmpeg` ใช้งานไม่ได้ ให้ติดตั้ง FFmpeg และเพิ่มลง `PATH` ก่อน
- ถ้า frontend เรียก backend ไม่ได้ ให้ตรวจว่า `NEXT_PUBLIC_API_BASE` และ `ALLOWED_ORIGINS` ตั้งค่าให้ตรงกับพอร์ตจริง
- ถ้า PowerShell ไม่อนุญาตให้ activate virtual environment ให้เปิด shell ใหม่หรือใช้วิธี activate ที่เหมาะกับเครื่องของคุณ
- หากเปลี่ยนพอร์ต backend จาก `8000` ต้องอัปเดตค่า `NEXT_PUBLIC_API_BASE` ให้ตรงกันด้วย
