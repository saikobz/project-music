# HarmoniQ – AI Music Stem & Mastering Toolkit

เว็บแอปสำหรับแยกเสียงจากไฟล์ WAV ด้วย AI พร้อม EQ, Compressor, Pitch Shift, และเครื่องมือวิเคราะห์คีย์/เทมโป/โทนเสียง มีเครื่องเล่นหลายสเตมบนหน้าเว็บให้ลองฟังและดาวน์โหลดรายแทร็ก

## คุณสมบัติหลัก
- แยกเสียงด้วย Open-Unmix (vocals, drums, bass, other) แล้วแพ็กเป็น ZIP
- ใส่ EQ แบบ preset ตามสเตม, Compressor หลายระดับ, และ Pitch Shift
- Auto-EQ (AI) ด้วยโมเดล `autoeq_cnn_v1.pt`
- วิเคราะห์เพลง: Tempo, Key, Median Pitch (โน้ต)
- ตัวเล่นเสียงแบบ Multi-Stem (ควบคุม mute/play/seek ต่อแทร็ก) และตัวเล่นไฟล์เดี่ยวแบบ waveform

## ความต้องการระบบ
- Python 3.10 (64-bit), Node.js 20+
- FFmpeg ต้องอยู่ใน PATH
- GPU + CUDA (ถ้ามี) จะช่วยให้ Open-Unmix ทำงานเร็วขึ้น

## ตั้งค่า Backend (FastAPI)
```bash
cd backend
py -3.10 -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```
- ค่าอัปโหลดสูงสุด 100MB ต่อไฟล์ (`.wav` เท่านั้น)
- ตัวแปรแนะนำ:
  - `ALLOWED_ORIGINS` (ค่าเริ่มต้น `http://localhost:3000`)
  - `SEPARATE_TTL_SECONDS` อายุไฟล์/โฟลเดอร์ที่สร้างก่อนตั้งเวลาลบ (ดีฟอลต์ 6 ชม.)
- เอาต์พุตหลัก: `uploads/` (ไฟล์ต้นฉบับ/ZIP), `separated/` (ไฟล์สเตม), `eq_applied/` (รวม Auto-EQ), `compressed/`

## ตั้งค่า Frontend (Next.js 15)
```bash
cd C:\Users\nopma\Desktop\project-music
npm install
npm run dev          # โหมดพัฒนา (พอร์ต 3000)
npm run build        # สร้าง production
npm start            # รัน production หลัง build แล้ว
```
- ตั้ง API base ได้ที่ `NEXT_PUBLIC_API_BASE` (ดีฟอลต์ `http://localhost:8000`)

## Health Check / API สำคัญ
```
curl http://localhost:8000/health    # ควรได้ {"status": "ok"}
POST /apply-eq-ai (multipart "file")  # Auto-EQ ด้วยโมเดล CNN
```

## หมายเหตุ
- ไฟล์ที่สร้างจะถูกตั้ง timer ลบตาม `SEPARATE_TTL_SECONDS` เพื่อลดการใช้ดิสก์
- หากปรับพอร์ตหรือโดเมน backend ให้ตั้ง `NEXT_PUBLIC_API_BASE` และ CORS (`ALLOWED_ORIGINS`) ให้ตรงกัน
