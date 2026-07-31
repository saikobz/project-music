# HarmoniQ — เอกสารสถาปัตยกรรมและการออกแบบระบบ (Architecture & Design)

> เอกสารฉบับนี้สรุปสถาปัตยกรรมโดยรวมของโปรเจกต์ HarmoniQ — ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI
> อัปเดตล่าสุด: 2026-08-01

---

## 1. ภาพรวมโปรเจกต์ (Project Overview)

HarmoniQ เป็นเว็บแอปพลิเคชันสำหรับ:
- **แยกแทร็กเสียงดนตรี** (Stem Separation) — แยกเพลงออกเป็น 4 แทร็ก: Vocals, Drums, Bass, Other ด้วย Open-Unmix
- **ปรับแต่งเสียงด้วย AI** — Auto-EQ (CNN/LSTM), Compressor, Pitch Shift, LUFS Mastering
- **วิเคราะห์เสียง** — Tempo, Key, Pitch ด้วย librosa

### สถาปัตยกรรมไฮบริด

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│  Frontend (Next.js :3000)   │      │  Backend (FastAPI :8000)     │
│  React 19 / TS / Tailwind 4 │ HTTP │  Python 3.10 / PyTorch       │
│  NextAuth v4                │─────▶│  Open-Unmix / librosa        │
│  WaveSurfer.js              │      │  Pedalboard / Omise          │
│       │                     │      │       │                      │
│       ▼                     │      │       ▼                      │
│  SQLite (Prisma)            │      │  uploads/ separated/         │
│  dev.db                     │      │  eq_applied/ compressed/     │
└─────────────────────────────┘      └──────────────────────────────┘
```

- **Frontend** เรียกใช้ **Backend API** ผ่านพอร์ต `8000` (ควบคุมด้วย `NEXT_PUBLIC_API_BASE`)
- **Database (SQLite)** ถูกใช้งานโดย Frontend เท่านั้นผ่าน Prisma ORM
- **Backend** เก็บไฟล์เสียงชั่วคราวบนดิสก์ (ถูก cleanup ลบอัตโนมัติตาม TTL)

---

## 2. Tech Stack

### Frontend
| เทคโนโลยี | เวอร์ชัน | บทบาท |
|-----------|---------|--------|
| Next.js (App Router) | 15 | Framework หลัก, SSR/CSR, API Routes |
| React | 19 | UI Library |
| TypeScript | — | ตรวจสอบชนิดข้อมูล |
| Tailwind CSS | 4 | ระบบ UI (Dark theme #0A0A0A) |
| NextAuth | v4 | Authentication (JWT strategy) |
| WaveSurfer.js | v7 | Waveform player (multi-track sync) |
| axios / fetch | — | HTTP client |
| sonner | — | Toast notifications |
| lucide-react | ^1.26 | Icons |

### Backend
| เทคโนโลยี | เวอร์ชัน | บทบาท |
|-----------|---------|--------|
| FastAPI | 0.115 | Web framework, routing |
| uvicorn | 0.34 | ASGI server |
| PyTorch / torchaudio | 2.7 | Deep learning (Open-Unmix, AutoEQ) |
| openunmix | 1.2.1 | Stem separation (UMXL) |
| librosa | 0.11 | Audio analysis, pitch shift, mel-spectrogram |
| soundfile | 0.13 | WAV read/write |
| pedalboard | 0.9.8 | DSP effects (Compressor, Filter, Limiter) |
| pyloudnorm | 0.1.1 | LUFS loudness measurement |
| pydub | 0.25 | WAV→MP3 conversion (320kbps) |
| pytest | 8.2 | Testing |

### Database & Payment
| เทคโนโลยี | บทบาท |
|-----------|--------|
| SQLite + Prisma ORM v6 | ฐานข้อมูล (`file:./dev.db`) |
| Omise | ระบบชำระเงิน (PromptPay QR + Credit Card แบบรายเดือน) |

---

## 3. โครงสร้างโฟลเดอร์

```
project-music/
├── app/                        # Next.js App Router
│   ├── page.tsx                # Landing page
│   ├── studio/                 # หน้า workspace ประมวลผลเสียง
│   ├── pricing/                # หน้าแพ็กเกจ (Free/Basic/Pro)
│   ├── api-pricing/            # หน้า pricing สำหรับ API developer
│   ├── auth/signin/            # หน้าเข้าสู่ระบบ/สมัครสมาชิก
│   ├── account/                # หน้าบัญชีผู้ใช้ (6 tabs + confirm-delete)
│   ├── dashboard/history/      # หน้าประวัติการประมวลผล
│   ├── about/ guide/ models/ support/ terms/ privacy/
│   ├── api/                    # API routes (Next.js)
│   │   ├── auth/               # NextAuth + register
│   │   ├── account/            # profile, password, preferences, providers, export
│   │   ├── history/            # project history CRUD
│   │   ├── subscription/       # checkout, cancel, history
│   │   ├── quota/              # consume, refund
│   │   └── webhooks/omise/     # payment webhook
│   └── components/             # UI components
├── lib/                        # Core libraries
│   ├── auth.ts                 # NextAuth config
│   ├── config.ts               # Constants (actions, genres, API_BASE)
│   ├── prisma.ts               # PrismaClient singleton
│   ├── subscription.ts         # Tier logic (quota, price, effective tier)
│   ├── download.ts             # downloadViaBlob helper
│   ├── omise.ts                # Omise client + https patch
│   ├── rate-limit.ts           # In-memory rate limiter
│   └── hooks/useAudioProcessor.ts  # Business logic กลางของ studio
├── prisma/
│   ├── schema.prisma           # Database schema (7 models)
│   └── migrations/             # Migration files
├── backend/                    # FastAPI backend
│   ├── main.py                 # FastAPI app, CORS, lifespan
│   ├── config.py               # Directory + env config
│   ├── cleanup_task.py         # TTL cleanup background task
│   ├── process_audio.py        # separate_audio, pitch_shift, analyze
│   ├── eq_compressor.py        # Compressor DSP
│   ├── auto_eq_inference.py    # Auto-EQ CNN/LSTM models
│   ├── auto_mastering.py       # Vocal polish + LUFS mastering
│   ├── models/                 # PyTorch model checkpoints
│   ├── routers/
│   │   ├── stems.py            # /separate, /download, /karaoke, /export
│   │   └── audio_ops.py        # /apply-eq-ai, /apply-compressor, /pitch-shift
│   ├── services/
│   │   ├── storage.py          # save_upload, convert_to_mp3
│   │   └── job_manager.py      # In-memory job session tracker
│   └── utils/auth_guard.py     # Tier + quota enforcement
├── tests/                      # Jest tests (frontend)
└── backend/tests/              # pytest (backend)
```

---

## 4. Database Schema (Prisma)

### ความสัมพันธ์ของ Model

```
User (1) ── (0..1) Subscription          [userId unique]
User (1) ── (N)   Account                [OAuth, cascade delete]
User (1) ── (N)   Session                [cascade delete]
User (1) ── (N)   UsageQuota             [unique(userId, periodStart)]
User (1) ── (N)   PaymentRecord          [cascade delete]
User (1) ── (N)   ProjectRecord          [index(userId)]
```

### 1. `User`
| Field | Type | หมายเหตุ |
|-------|------|----------|
| id | String @id | cuid() |
| name / email / image | String? | email unique |
| password | String? | bcrypt hash (null สำหรับ OAuth-only) |
| emailVerified | DateTime? | |
| omiseCustomerId | String? | สำหรับชำระเงิน |
| theme | String | DARK/LIGHT (default DARK) |
| language | String | TH/EN (default TH) |
| emailNotifications | Boolean | default true |

### 2. `Account` — OAuth accounts (Google/Facebook/LINE)
- `@@unique([provider, providerAccountId])`
- เก็บ access/refresh token สำหรับ reconnect

### 3. `Subscription`
| Field | Type | หมายเหตุ |
|-------|------|----------|
| userId | String @unique | 1:1 กับ User |
| tier | String | `FREE` / `BASIC` / `PRO` |
| status | String | `ACTIVE` / `PENDING` / `PAST_DUE` / `EXPIRED` / `CANCELED` |
| paymentMethod | String? | `CREDIT_CARD` / `PROMPTPAY` |
| omiseScheduleId | String? | เก็บ schedule รายเดือนของ Omise |
| currentPeriodStart/End | DateTime | รอบบิล |

### 4. `UsageQuota`
| Field | Type | หมายเหตุ |
|-------|------|----------|
| monthlyQuota | Int | FREE=3, BASIC=15, PRO=-1 (unlimited) |
| usedCount | Int | default 0 |
| periodStart / periodEnd | DateTime | รอบ 30 วัน |

- `@@unique([userId, periodStart])` — กันการสร้าง period ซ้ำจาก concurrent requests (F4)

### 5. `PaymentRecord`
| Field | Type | หมายเหตุ |
|-------|------|----------|
| omiseChargeId | String @unique | กัน webhook ซ้ำ |
| amount | Int | หน่วยสตางค์ |
| status | String | successful / failed / expired |
| paidAt | DateTime | |

### 6. `ProjectRecord` (ประวัติการประมวลผล)
| Field | Type | หมายเหตุ |
|-------|------|----------|
| action | String | `separate` / `apply-eq-ai` / `apply-compressor` / `pitch-shift` |
| originalFilename | String | ชื่อไฟล์ต้นฉบับ |
| fileId | String? | file_id ของ backend (เฉพาะ action ที่มีไฟล์ output) |
| stems | String? | JSON string เช่น `["Vocals","Drums","Bass","Other"]` |
| **expiresAt** | DateTime? | เวลาหมดอายุของไฟล์ = createdAt + TTL (เฉพาะที่มี fileId) |
| createdAt | DateTime | |

---

## 5. Frontend Architecture

### 5.1 Page Routes (14 routes)

| Route | Type | Auth | รายละเอียด |
|-------|------|------|-----------|
| `/` | Client | — | Landing page (Hero, features, How It Works, FAQ) |
| `/studio` | Client | Guest OK | หน้า workspace หลัก: upload + 4 actions |
| `/pricing` | Client | — | แพ็กเกจ 3 ระดับ + CheckoutModal |
| `/api-pricing` | Server | — | เอกสาร API สำหรับ developer |
| `/auth/signin` | Client | — | เข้าสู่ระบบ/สมัคร (Credentials + 3 OAuth) |
| `/account` | Client | ✓ | 6 tabs: Profile, Password, Connected Accounts, Preferences, Data, Billing |
| `/account/confirm-delete` | Client | ✓ | Re-auth ผ่าน OAuth ก่อนลบบัญชี/ยกเลิกสมาชิก |
| `/dashboard/history` | Client | ✓ | ประวัติ + สถานะไฟล์ + Play/Download/Delete |
| `/about`, `/guide`, `/models` | Server | — | หน้าสารคดี |
| `/support`, `/terms`, `/privacy` | Client | — | หน้าข้อมูล |

### 5.2 Components (app/components/)

**Layout:** `Navbar` (nav + UserMenu), `Footer`, `SessionProvider`

**Studio:**
- `UploadBox` — ฟอร์มหลัก (~660 บรรทัด): drag-drop, 4 tabs action, config panels, trim, progress
- `WaveformPlayer` — single-file player (WaveSurfer.js)
- `AdvancedMultiTrackPlayer` — 4-stem ซิงค์พร้อมกัน (mute/solo/volume/polish)
- `MultiStemLivePlayer` — adapter ส่ง fileId → baseUrl
- `AudioAnalysis` — แสดง Tempo/Key/Pitch
- `ExportMasterModal` — export mix/stems + LUFS target
- `SingleExportModal` — export single file (แปลง format)

**Auth/อื่น ๆ:** `CheckoutModal` (PromptPay/CC), `UserMenu`, `HowItWorks`, `FaqSection`

**Settings:** `AutoEqSettings`, `CompressorSettings`, `PitchShiftSettings`

### 5.3 API Routes (Next.js)

| กลุ่ม | Route | หน้าที่ |
|-------|-------|--------|
| Auth | `api/auth/[...nextauth]` | NextAuth handler |
| | `api/auth/register` | สมัคร Credentials + สร้าง Subscription/Quota |
| Account | `api/account` (GET/DELETE) | ข้อมูลผู้ใช้ + ลบบัญชี |
| | `api/account/profile` (PUT) | แก้ไข name/email |
| | `api/account/password` (PUT) | เปลี่ยนรหัสผ่าน |
| | `api/account/preferences` (PUT) | theme/language/notifications |
| | `api/account/providers` (GET/DELETE) | จัดการ OAuth |
| | `api/account/export` (GET) | export ข้อมูลผู้ใช้เป็น JSON |
| History | `api/history` (GET/POST) | รายการประวัติ + สร้าง record |
| | `api/history/[id]` (DELETE) | ลบ record (atomic + ownership) |
| Subscription | `api/subscription/checkout` (POST) | ชำระเงิน (PromptPay/CC) |
| | `api/subscription/cancel` (POST) | ยกเลิกสมาชิก |
| | `api/subscription/history` (GET) | ประวัติการชำระเงิน |
| Quota | `api/quota/consume` (POST) | หักโควตา (atomic) |
| | `api/quota/refund` (POST) | คืนโควตาเมื่อ processing ล้มเหลว |
| Webhook | `api/webhooks/omise` (POST) | รับการแจ้งเตือนการชำระเงิน |

### 5.4 Core Libraries (lib/)

| ไฟล์ | บทบาท |
|------|--------|
| `auth.ts` | NextAuth config: 4 providers, JWT, callbacks (tier enrichment), `requireSession()`, `verifyAccountAuth()` |
| `config.ts` | `API_BASE_URL`, `MAX_UPLOAD_BYTES`, `ACTION_TO_BACKEND`, `AUDIO_ACTIONS` |
| `prisma.ts` | PrismaClient singleton (กัน hot-reload ซ้ำ) |
| `subscription.ts` | `TIER_MONTHLY_QUOTA`, `TIER_PRICES`, `getEffectiveTier()` |
| `download.ts` | `downloadViaBlob()` — fetch→blob→anchor (กันปัญหา cross-origin download) |
| `omise.ts` | Omise client + patch `https.request` (กันขัดกับ openid-client) |
| `rate-limit.ts` | In-memory rate limiter (10 req/min/IP) |
| `hooks/useAudioProcessor.ts` | **หัวใจของ studio** — จัดการ upload, 4 actions, quota, history, export, karaoke, analysis |

---

## 6. Backend Architecture

### 6.1 Endpoints

**Router `stems.py` (tags: stems):**
| Method | Path | รายละเอียด |
|--------|------|-----------|
| POST | `/separate` | แยกเสียง 4 stems → สร้าง ZIP → คืน `{file_id, zip_url}` |
| GET | `/download/{file_id}` | **Generic download** — หา ZIP ก่อน แล้วค้นหาไฟล์ขึ้นต้น `{file_id}_` ใน uploads/, eq_applied/, compressed/ |
| GET | `/separated/{file_id}/{filename}` | ไฟล์ stem เดี่ยว (ใช้กับ player) |
| GET | `/karaoke/{file_id}` | รวม Drums+Bass+Other เป็น backing track |
| POST | `/api/process/vocal-polish` | ขัดเกลาเสียงร้อง (De-esser + Compressor + Air EQ) |
| POST | `/api/process/export` | Export mixdown/stems แบบคัสตอม + LUFS mastering |

**Router `audio_ops.py` (tags: audio_ops):**
| Method | Path | รายละเอียด |
|--------|------|-----------|
| POST | `/apply-eq-ai` | Auto-EQ AI (CNN/LSTM) → คืน `X-File-Id` header |
| POST | `/apply-compressor` | Compressor (genre presets + manual overrides) → คืน `X-File-Id` |
| POST | `/pitch-shift` | ปรับระดับเสียง ±semitones → คืน `X-File-Id` |
| POST | `/convert-format` | แปลง WAV/MP3 (ไม่หักโควตา) |
| POST | `/analyze` | วิเคราะห์ tempo/key/pitch |

**main.py:** `GET /health`, lifespan (cleanup task), CORS (`expose_headers=["X-File-Id"]`), global exception handler

### 6.2 Processing Pipeline

```
Client POST /separate
  → validate_request_quota()        # ตรวจ tier + โควตา (B6: ก่อนประมวลผล)
  → save_upload()                   # save ไฟล์ + validate (.wav, ≤100MB) + trim
  → increment_guest_quota()         # นับโควตา guest หลังผ่าน validation
  → job_manager.register_job()      # ลงทะเบียน job
  → processing_semaphore            # จำกัด concurrent = MAX_CONCURRENT_TASKS (2)
  → separate_audio()                # Open-Unmix → vocals/drums/bass/other
  → convert_to_mp3()                # ถ้า export_format=mp3
  → create_zip_archive()            # สร้าง ZIP
  → return {file_id, zip_url}
  → finally: os.remove(input_path)  # ลบไฟล์ input ทันที
```

### 6.3 AI Models

| โมเดล | ไฟล์ | รายละเอียด |
|-------|------|-----------|
| Open-Unmix UMXL | `backend/models/umxl/` | 4 targets (vocals/drums/bass/other), GPU support, lru_cache |
| AutoEQ CNN | `autoeq_cnn_v1.pt` | 3 Conv layers → EQ gain curve (genre-based anchors) |
| AutoEQ LSTM | `autoeq_lstm_last.pt` | Bi-LSTM 2 ชั้น + genre embedding, blend 0.65 กัน over-correction |

### 6.4 Tier Enforcement (auth_guard.py)

| Tier | โควตาต่อเดือน | Max Pitch Shift | CNN Auto-EQ |
|------|---------------|-----------------|-------------|
| FREE | 3 | ±2 semitones | ❌ (LSTM เท่านั้น) |
| BASIC | 15 | ±6 semitones | ✅ |
| PRO | Unlimited | ±12 semitones | ✅ |

- **Guest quota:** เก็บในไฟล์ JSON (per-IP, reset รายวัน, thread-safe + atomic write)
- **หลักการ B6:** validate ก่อนหักโควตา — ไฟล์ไม่ผ่าน validation ไม่เสียโควตา
- **หลักการ B1:** guest ถูกบังคับเป็น FREE เสมอ (ignore `X-User-Tier` header ที่สปูฟ)

### 6.5 File Lifecycle & TTL Cleanup

```
upload → {uuid}_{filename}.wav  ใน uploads/
   ↓ ประมวลผล
output ไฟล์ใน separated/ | eq_applied/ | compressed/
   ↓ ทุก 5 นาที (CLEANUP_INTERVAL_SECONDS=300)
periodic_cleanup() ลบไฟล์ที่ mtime เก่ากว่า 20 นาที (SEPARATE_TTL_SECONDS=1200)
```

- ลบทั้งไฟล์และโฟลเดอร์ (`shutil.rmtree`)
- ไฟล์ input ถูกลบ **ทันที** หลังประมวลผล (finally block)
- Frontend รู้เวลาหมดอายุผ่าน `expiresAt` ใน `ProjectRecord` (คำนวณจาก TTL เดียวกัน)

---

## 7. Flow สำคัญ (Key Flows)

### 7.1 Authentication

```
User ลงทะเบียน (Credentials)
  → POST /api/auth/register
  → สร้าง User + Subscription(FREE) + UsageQuota(3) ใน transaction เดียว
  → auto sign-in

User เข้าสู่ระบบ (OAuth)
  → signIn callback: upsert Subscription + UsageQuota (กัน race)
  → jwt callback: stamp tier, omiseCustomerId, reauthAt
  → session callback: อ่าน tier ล่าสุดจาก DB ทุกครั้ง (กัน tier ค้าง — L1)
```

### 7.2 Subscription & Payment

```
Checkout (Credit Card):
  → destroy schedule เก่า (F3 — กัน double billing)
  → สร้าง/update Omise customer + card (F7)
  → สร้าง monthly schedule (metadata: {userId, tier})
  → Subscription status = PENDING (F6 — ยังไม่ได้สิทธิ์จนกว่าจ่ายสำเร็จ)

Webhook (charge.complete / schedule.process):
  → ตรวจสอบความถูกต้อง (HMAC หรือ event fetch-back — fail-closed)
  → activateSubscription(): status=ACTIVE, ลบ UsageQuota เก่า (L8), สร้างใหม่

Cancel:
  → verifyAccountAuth (password หรือ OAuth re-auth ภายใน 5 นาที)
  → destroy schedule, status=CANCELED
  → ผู้ใช้ยังใช้สิทธิ์ tier เดิมจนจบรอบบิล
```

### 7.3 Quota Consumption (TOCTOU-safe)

```typescript
// อัปเดตแบบ atomic — กัน 2 requests พร้อมกันผ่านทั้งคู่ (F4)
prisma.usageQuota.updateMany({
  where: { id, OR: [{ monthlyQuota: -1 }, { usedCount: { lt: max } }] },
  data: { usedCount: { increment: 1 } },
});
```

- PRO (-1) ผ่านเสมอ / คนอื่นต้องมี `usedCount < max`
- `getEffectiveTier()`: status ≠ ACTIVE หรือเกิน period → ถือเป็น FREE
- Refund: ลดแบบ conditional (`usedCount > 0`) — กันติดลบจากการ refund ซ้ำ

### 7.4 History & Expiry (ประวัติ + หมดอายุ)

```
Processing สำเร็จ
  → backend คืน X-File-Id (ทุก action ที่มีไฟล์ output)
  → saveHistory(action, fileId)  — fire-and-forget (ไม่ await)
  → POST /api/history: ตั้ง expiresAt = now + SEPARATE_TTL_SECONDS (1200s)

หน้า /dashboard/history
  → GET /api/history (เติม expiresAt ให้ record เก่าจาก createdAt + TTL)
  → UI แสดงสถานะ: "หมดอายุ 15:42 น." (amber) หรือ "หมดอายุ" (red, ปุ่มปิด)
  → Play: separate→/separated/{id}/vocals.wav, อื่นๆ→/download/{id}
  → Download: /download/{id} (search ทุก dir)
  → Delete: window.confirm + DELETE /api/history/{id}
```

---

## 8. Design Decisions ที่สำคัญ

| # | การตัดสินใจ | เหตุผล |
|---|-------------|--------|
| 1 | JWT session + อ่าน tier จาก DB ทุก request | กันข้อมูล subscription ค้างหลังอัปเกรด/ยกเลิก (L1) |
| 2 | History เป็น fire-and-forget | ไม่รบกวน UX หลัก ถ้าบันทึกประวัติล้มเหลว |
| 3 | Dual quota: DB (login) + JSON per-IP (guest) | Guest ไม่มี account ก็จำกัดการใช้ได้ |
| 4 | Credit Card → status PENDING ก่อน | สิทธิ์จะ activate เมื่อจ่ายสำเร็จเท่านั้น (F6) |
| 5 | `expiresAt` เก็บใน DB (ไม่คำนวณฝั่ง UI) | Frontend รู้ TTL เดียวกันกับ backend cleanup |
| 6 | Generic `/download/{file_id}` search ทุก dir | ครอบคลุมทุก action โดยไม่ต้องสร้าง endpoint แยก |
| 7 | `X-File-Id` header + `expose_headers` CORS | เก็บ file_id ได้จาก blob response ของ EQ/Compressor/Pitch |
| 8 | Static expiry time (`หมดอายุ 15:42 น.`) | ไม่ต้อง setInterval re-render ทุกวินาที (optimize) |
| 9 | Semaphore `MAX_CONCURRENT_TASKS=2` | จำกัดโหลด GPU/CPU ตอนประมวลผลหนัก |
| 10 | Webhook fail-closed (HMAC/event verify) | กันการปลอม webhook — ไม่มี secret = reject หมด |
| 11 | Atomic quota `updateMany` conditional | กัน TOCTOU race — 2 requests พร้อมกันผ่านได้แค่ 1 (F4) |
| 12 | `deleteMany` + ownership check | กันลบ record ของคนอื่น (L6) |

---

## 9. Environment Variables

| ตัวแปร | Default | ฝั่ง | รายละเอียด |
|--------|---------|-----|-----------|
| `DATABASE_URL` | `file:./dev.db` | Frontend | Prisma SQLite path |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Frontend | Backend API URL |
| `NEXTAUTH_SECRET` | — | Frontend | NextAuth encryption key |
| `NEXTAUTH_URL` | `http://localhost:3000` | Frontend | Canonical URL |
| `GOOGLE_CLIENT_ID/SECRET` | — | Frontend | Google OAuth |
| `FACEBOOK_CLIENT_ID/SECRET` | — | Frontend | Facebook OAuth |
| `LINE_CLIENT_ID/SECRET` | — | Frontend | LINE OAuth |
| `OMISE_SECRET_KEY` | — | Frontend | Omise secret key |
| `NEXT_PUBLIC_OMISE_PUBLIC_KEY` | — | Frontend | Omise public key |
| `OMISE_WEBHOOK_SECRET` | — | Frontend | Webhook HMAC secret |
| `SEPARATE_TTL_SECONDS` | `1200` | ทั้งคู่ | TTL ก่อนลบไฟล์ (20 นาที) |
| `CLEANUP_INTERVAL_SECONDS` | `300` | Backend | ความถี่ cleanup (5 นาที) |
| `MAX_CONCURRENT_TASKS` | `2` | Backend | งานประมวลผลพร้อมกันสูงสุด |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Backend | CORS allowed origins (comma-separated) |

---

## 10. Testing

### Frontend (Jest — 19 suites, 116 tests)
- **auth.test.ts** — anti-enumeration, session tier refresh (L1)
- **register.test.ts** — validation, duplicate email
- **checkout.test.ts** — PromptPay/CC flows, destroy old schedule (F3), PENDING (F6)
- **webhook.test.ts** — HMAC/event verification, fail-closed
- **quota-consume.test.ts** — TOCTOU race (F4), canceled→FREE (F2)
- **quota-refund.test.ts** — idempotent, no negative
- **history.test.ts** — expiresAt enrich + set, ownership delete (L6)
- **account.test.ts** — re-auth, lockout protection (M20)
- **omise.test.ts** — https.request patch (L5)
- **download.test.ts** — downloadViaBlob (M9)
- **upload-box.test.tsx** — quota 403, refund on fail, trim validation
- **advanced-multi-track-player.test.tsx** — WaveSurfer lifecycle (M1-M4/M6)

### Backend (pytest — 22 tests)
- **test_auth_guard.py** — tier limits, guest quota (per-IP, daily reset, atomic save), B1/B6/B7 rules

---

## 11. Security Notes

- **Rate limiting:** `/api/auth/callback/credentials`, `/api/auth/register`, `/api/account/password` → 10 req/min/IP
- **Anti-enumeration:** dummy bcrypt hash + error message เดียวกัน (F9)
- **Re-auth สำหรับ destructive actions:** เปลี่ยน email / ยกเลิกสมาชิก / ลบบัญชี → ต้องยืนยันตัวตนใหม่ (M19+)
- **Lockout protection:** ไม่ให้ unlink OAuth provider สุดท้ายถ้าไม่มี password (M20)
- **Global exception handler:** log ฝั่ง server เท่านั้น, client ได้ข้อความ generic (M17)
