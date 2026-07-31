# 📋 งานที่ยังเหลือ — HarmoniQ (Action Items ภายนอก + รายการที่เลื่อน)

> อัปเดตล่าสุด: 2026-07-31
> หมายเหตุ: งานใน codebase ทั้งหมด (Bugs/Critical/High/Medium/Low/C-items) แก้เสร็จแล้ว
> ยกเว้นรายการด้านล่าง ซึ่งส่วนใหญ่เป็น **งานภายนอก/DevOps** ที่ทำในโค้ดไม่ได้

---

## 🏗️ ส่วนที่ 1: Action Items ภายนอก (ต้องทำก่อน/ตอน Deploy — เรียงตามความสำคัญ)

### 1. ตั้งค่า Omise Keys (G1/G3) ✅ env ครบแล้ว — เหลือตั้ง Webhook URL + ทดสอบ Sandbox

| รายละเอียด | ค่า |
|------------|-----|
| **สิ่งที่ทำแล้ว** | 1) `OMISE_WEBHOOK_SECRET` (64 chars) ตั้งแล้วใน `.env`<br>2) `OMISE_SECRET_KEY` + `NEXT_PUBLIC_OMISE_PUBLIC_KEY` (test keys) กรอกแล้ว<br>3) Webhook logic รองรับ 2 พฤติกรรม: มี signature → ตรวจ HMAC / ไม่มี signature → `omise.events.retrieve` fallback |
| **ยังต้องทำ** | 1) ตั้ง Webhook URL ใน Omise Dashboard (ชี้ `/api/webhooks/omise`)<br>2) ทดสอบกับ Sandbox จริง: charge.complete → subscription ACTIVE; event ปลอม → 403 |
| **ตรวจสอบ** | POST ทดสอบ event ปลอม → 403; event จริงจาก Sandbox → 200 + upsert subscription |
| **สถานะ** | 🟡 ทำเกือบครบ — เหลือตั้ง Webhook URL + ทดสอบ Sandbox จริง |

### 2. ติดตั้ง `ffmpeg` (ทั้งเครื่องพัฒนาและ production) ✅ ติดตั้งแล้ว (เครื่อง dev)

| รายละเอียด | ค่า |
|------------|-----|
| **ปัญหา** | การแปลง MP3 ทุกจุดใช้ `pydub` ซึ่งต้องพึ่ง ffmpeg — **เครื่องนี้ยังไม่มี ffmpeg** → export MP3, `/convert-format`, stems MP3 จะ fail ที่ runtime |
| **วิธีทำ** | - Windows: `winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements` ✅ **ทำแล้ว (v8.1.2)**<br>- Linux: `apt install ffmpeg`<br>- macOS: `brew install ffmpeg` |
| **ตรวจสอบ** | รัน `ffmpeg -version` ใน environment ที่ deploy — dev machine ผ่านแล้ว (72/72 tests ไม่มี warning) |
| **สถานะ** | ✅ เครื่อง dev ติดตั้งแล้ว — **production ยังต้องติดตั้ง** |

### 3. ทดสอบ Webhook กับ Omise Sandbox ⚠️ ควรทำก่อนเปิดจริง

| รายละเอียด | ค่า |
|------------|-----|
| **ปัญหา** | เรา implement signature ตรวจแบบ HMAC-SHA256 บน raw body + event keys (`charge.complete`, `schedule.process` ฯลฯ) ตามเอกสาร Omise — **ยังไม่เคยยืนยันกับของจริง** ถ้า Omise ใช้ format ต่าง (เช่น ไม่ส่ง signature header หรือ event key ต่างชื่อ) webhook จะทำงานผิดเงียบๆ |
| **วิธีทำ** | 1) ตั้ง `OMISE_WEBHOOK_SECRET` + `OMISE_SECRET_KEY` (sandbox key) ใน env<br>2) เปิด Omise Dashboard (โหมด Test) → ตั้ง Webhook URL ชี้ไป production/staging<br>3) ทำ charge ทดสอบ (PromptPay + Card + schedule) แล้วดูว่า event มาถึงและถูกประมวลผลถูกต้อง<br>4) ถ้า Omise ไม่ส่ง HMAC signature → ต้องเปลี่ยนเป็นวิธี verify event id ย้อนกลับที่ Omise API (`omise.events.retrieve`) |
| **สถานะ** | ☐ ยังไม่ได้ทำ |

### 4. Rate Limit → Redis/Upstash (เฉพาะถ้า deploy แบบหลาย instance) ⚠️ ตามความจำเป็น

| รายละเอียด | ค่า |
|------------|-----|
| **ปัญหา** | `lib/rate-limit.ts` เก็บ bucket ในหน่วยความจำต่อ process — ถ้ารันหลาย instance (load balancer) แต่ละ instance นับแยกกัน → brute force ได้ ~N เท่า |
| **วิธีทำ** | เปลี่ยน backend ของ `checkRateLimit` จาก `Map` ในหน่วยความจำ → Redis (เช่น `@upstash/ratelimit` หรือ `ioredis`) |
| **สถานะ** | ☐ ยังไม่ได้ทำ (รอถ้ามีหลาย instance จริง) |

---

## 💻 ส่วนที่ 2: รายการใน codebase ที่เลื่อนออก (พร้อมเหตุผล)

### ~~1. C6 — FastAPI Exception Handler กลาง~~ ✅ แก้แล้ว

| รายละเอียด | ค่า |
|------------|-----|
| **สถานะ** | ✅ ทำแล้ว — `backend/main.py#unhandled_exception_handler` (global handler คืน 500 JSON generic) + ลบ generic `except Exception` ออกจาก 6 endpoints; เก็บ handlers เฉพาะ (ValueError→400, AutoEQModelLoadError→503, HTTPException) + `finally` cleanup ไว้ครบ; tests: `test_global_exception_handler.py` (2 tests) |

### 2. M21 (ส่วนต่อยอด) — Email Verification Flow เต็มรูปแบบ

| รายละเอียด | ค่า |
|------------|-----|
| **ปัญหา** | ปัจจุบันเปลี่ยน email ต้องยืนยันตัวตน (password/confirmEmail) แล้ว — แต่ยังไม่มีระบบส่งอีเมลยืนยันไปยัง **อีเมลใหม่** ก่อน activate |
| **วิธีทำเมื่อพร้อม** | เพิ่ม provider ส่งอีเมล (เช่น Resend/SES) + ตาราง verification token + ลิงก์ยืนยันก่อนเปลี่ยน email |
| **สถานะ** | ☐ เลื่อน (ต้องการ email infrastructure) |

### 3. M19 (ส่วนต่อยอด) — Re-authentication เต็มรูปแบบผ่าน OAuth ✅ ทำแล้ว

| รายละเอียด | ค่า |
|------------|-----|
| **สถานะ** | ✅ เสร็จ — JWT stamp `reauthAt` ตอน sign-in ใหม่ + session expose; DELETE account / cancel บังคับ OAuth-only user re-auth ผ่าน provider ภายใน 5 นาที (403 ถ้าเก่า); หน้าใหม่ `/account/confirm-delete` (ปุ่มยืนยันด้วย provider → กลับมา auto-execute); `DataSection`/`BillingSection` เปลี่ยนเป็น redirect ไปหน้า confirm สำหรับ OAuth user |

---

## ✅ สรุปสถานะปัจจุบัน (สำหรับอ้างอิง)

| หมวด | แก้แล้ว | ค้าง |
|-------|---------|------|
| Critical (B1-B5, F1-F3) | 8/8 | 0 |
| High | 20/20 | 0 |
| Medium (M1-M21) | 21/21 | 0 |
| Low (L1-L13) | 13/13 | 0 |
| Clean Code (C1-C16) | 16/16 | 0 |
| ต่อยอด (M19+ OAuth re-auth, M21+ email) | 1/2 | 1 (M21+ email flow) |
| **งานใน codebase** | **79/80** | **1** (email verification flow) |
| **Action ภายนอก** | 1/4 (ffmpeg dev) | **3** (Omise keys+Sandbox, ffmpeg prod, Redis) |

## 🧪 คำสั่งตรวจสอบหลังทำ Action Item แต่ละข้อ

```powershell
# 1. หลังตั้ง OMISE_WEBHOOK_SECRET + ติดตั้ง ffmpeg
.\.venv\Scripts\python.exe -m pytest backend/tests tests        # backend tests
npx jest                                                          # frontend tests
ffmpeg -version                                                   # ยืนยัน ffmpeg

# 2. หลังทดสอบ Sandbox เสร็จ
#    - ทดสอบ webhook จริง 1 รอบ: charge.complete -> subscription ACTIVE
#    - ทดสอบ export MP3 / convert-format จริง (ต้องใช้ ffmpeg)
```

## 🔗 วิธีตั้ง Webhook URL ใน Omise Dashboard (G3)

1. เข้า https://dashboard.omise.co → login → **เลือกโหมด Test**
2. **Settings → Webhooks → Add Webhook**
3. URL: `https://<โดเมน>/api/webhooks/omise`
4. เลือก events: `charge.complete`, `charge.failed`, `charge.expired`, `schedule.process`, `schedule.expired`
5. Save

**URL ต้องเป็นสาธารณะ** (Omise server ต้องเข้าถึงได้):
- มี staging/deploy → ใช้ URL จริง
- ทดสอบจากเครื่อง dev → ใช้ tunnel: `winget install ngrok` แล้ว `ngrok http 3000` → ตั้ง URL = `https://<ngrok>.ngrok.io/api/webhooks/omise`

**ตรวจสอบ**: ทำ charge ทดสอบ → ดู log `npm run dev` มี request มาที่ webhook → subscription เปลี่ยน ACTIVE

---

*ไฟล์นี้เป็นส่วนต่อจาก `CODE_REVIEW.md` (รายงานปัญหา + สถานะการแก้ไขทั้งหมด) และ `IMPLEMENTATION_PLAN.md` (แผนการดำเนินงาน)*
