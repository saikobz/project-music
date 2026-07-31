# 📋 รายงาน Code Review — HarmoniQ

**วันที่:** 2026-07-31
**ขอบเขต:** Backend (FastAPI/Python) + Frontend (Next.js/React 19/TS) + API Routes/Auth/Payment
**หมายเหตุ:** รายงานนี้เป็นผลการตรวจสอบแบบ Read-only ยังไม่มีการแก้ไขโค้ดใดๆ

---

## สรุปภาพรวม

| ระดับ | จำนวน | ผลกระทบ |
|-------|-------|---------|
| 🔴 Critical | 8 | ช่องโหว่ด้านความปลอดภัย/รายได้ และข้อมูลผู้ใช้ถูกทำลาย |
| 🟠 High | ~20 | Crash, Race condition, ฟีเจอร์หลักทำงานผิดพลาด |
| 🟡 Medium/Low | ~30 | Dead code, Memory leak, UX/Code quality |

**3 กลุ่มที่กระทบธุรกิจมากที่สุด:**
1. **รายได้รั่ว** — Guest ปลอม tier (B1) + Webhook fail-open (F1) + tier ไม่เคยถูกลด (F2) = payment/quota bypass ได้ทั้งระบบ
2. **ฟีเจอร์เสียเงินทำงานผิดเงียบๆ** — B4 (Pedalboard แกนผิด stereo ไม่ทำงาน), B5 (NaN จากไฟล์เงียบ)
3. **ทำลายข้อมูลผู้ใช้** — B3 (ไฟล์ stems หายก่อน TTL เมื่อ export MP3)

---

# ส่วนที่ 1: Bugs / Logic Errors / Edge Cases

## 🐍 Backend (FastAPI/Python)

### 🔴 Critical

| # | ปัญหา | ตำแหน่ง | รายละเอียด |
|---|-------|---------|-----------|
| B1 | ~~Guest ปลอม tier ใช้งานไม่จำกัด~~ ✅ แก้แล้ว TU-1 | ~~`backend/utils/auth_guard.py:119-122`~~ | แยกเป็น `validate_request_quota` (ตรวจสิทธิ์ ไม่หัก) + `increment_guest_quota` (นับหลัง save_upload ผ่าน) — Guest **บังคับ tier=FREE** เสมอ (ไม่ trust `X-User-Tier`), CNN lock/pitch limit ยังบังคับกับ guest |
| B2 | **Path Traversal (เขียนไฟล์นอกโฟลเดอร์)** ~~✅ แก้แล้ว TU-2~~ | `backend/routers/audio_ops.py:101` → `backend/eq_compressor.py:193` | ~~param `genre` ของ `/apply-compressor` ไม่มี pattern validation แต่ถูกฝังในชื่อไฟล์ output~~ — เพิ่ม `pattern` whitelist ทั้ง 2 endpoints + strict validate ใน `apply_compression` |
| B3 | **Export MP3 ลบไฟล์ Stem ต้นฉบับทิ้ง** ✅ แก้แล้ว TU-2 | `backend/services/storage.py:74-75` + `backend/routers/stems.py:47-52, 266-269` | `convert_to_mp3` เพิ่ม param `remove_source` — call site ของ stem export ส่ง `remove_source=False` → WAV ต้นฉบับอยู่จนกว่า TTL |
| B4 | ~~Pedalboard ประมวลผลผิดแกน~~ **⛔ ไม่ใช่ bug (พิสูจน์แล้ว)** | `backend/auto_mastering.py:14, 41` | ทดสอบ empirical กับ pedalboard 0.9.8: `(samples, channels)` จาก `sf.read` ถูก process ถูกต้อง (HighShelfFilter boost เฉพาะช่องความถี่สูง) — เวอร์ชันนี้ auto-detect layout; mono/stereo ทำงานถูกอยู่แล้ว มี regression tests กันไว้แล้ว |
| B5 | **NaN/garbage เมื่อเสียงเงียบสนิท** ✅ แก้แล้ว TU-3 | `backend/auto_mastering.py:27-34` | ~~`integrated_loudness` คืน `-inf` → `gain = inf` → `0*inf = NaN` → Limiter กลืน NaN เป็น full-scale DC 1.0 ทั้งไฟล์~~ — เพิ่ม `np.isfinite` guard + try/except → ข้าม gain stage ไฟล์เงียบคงเงียบ |

### 🟠 High

| # | ปัญหา | ตำแหน่ง | รายละเอียด |
|---|-------|---------|-----------|
| B6 | ~~โควตาถูกหักก่อน validate/ประมวลผล ไม่มีคืนเมื่อล้มเหลว~~ ✅ แก้แล้ว TU-1 | ~~`backend/utils/auth_guard.py:117-122` + routers ทุกตัว~~ | `validate_request_quota` เรียกก่อนงาน (ไม่หัก) → `save_upload` ผ่าน → `increment_guest_quota` (หักเฉพาะ guest หลังไฟล์ valid) — ไฟล์ผิดนามสกุล/ใหญ่เกินไม่เสียโควตาอีกต่อไป |
| B7 | ~~ไฟล์โควตา JSON เสีย → รีเซ็ตโควตาทุก IP เงียบๆ + race~~ ✅ แก้แล้ว TU-1 | ~~`backend/utils/auth_guard.py:38-39, 43-49`~~ | เขียนแบบ atomic (`tmp` + `os.replace`) + `threading.Lock` คุ้ม read-modify-write; ไฟล์เสีย log error แทนเงียบ |
| B8 | **Trim ไม่ validate → slice ติดลบ/ไฟล์ว่าง** | `backend/services/storage.py:53-62` | ไม่เช็ค `trim_start >= 0`, `start < end`, `end <= duration` → `trim_start=-5` ทำ numpy wrap-around (ผิดเพลง), `start > end` ได้ array ว่าง → `sf.write` ไฟล์ 0 sample → librosa/openunmix crash ใน step ถัดไป (500) |
| B9 | **`convert_to_mp3` ล้มเหลวเงียบ คืน path เดิม** | `backend/services/storage.py:77-79` | เมื่อ ffmpeg/pydub พัง ฟังก์ชัน catch แล้ว**คืน path WAV เดิม** → caller ส่งไฟล์ WAV พร้อม `media_type="audio/mpeg"` และใน zip stems ไฟล์ `.wav` ถูกตั้งชื่อ `.mp3` (`stems.py:268`) |

### 🟡 Medium / Low

| # | ปัญหา | ตำแหน่ง | รายละเอียด |
|---|-------|---------|-----------|
| B10 | zipfile สร้างแบบ synchronous บน event loop | `backend/routers/stems.py:56-61` | หลายร้อย MB block request อื่นทั้งหมด (ใน `process_export` ใช้ `asyncio.to_thread` แล้ว แต่ `/separate` ไม่) |
| B11 | `_jobs` dict ไม่เคย evict | `backend/services/job_manager.py:16` | Memory leak สะสมทุก file_id จน restart |
| B12 | karaoke/custom mix สมมติ sample rate/ช่องเสียงเท่ากันหมด | `backend/routers/stems.py:152-156, 240-243` | ไม่มีการ resample/reshape → mono+stereo ปนกัน = broadcast error 500 |
| B13 | `/analyze` ไม่มี quota check เลย | `backend/routers/audio_ops.py:196-214` | ใครก็ยิง librosa (CPU หนัก) ฟรีไม่จำกัด |
| B14 | `_cleanup_partial_checkpoints()` เป็น dead code | `backend/process_audio.py:23` | ไม่เคยถูกเรียก (ตั้งใจให้ช่วยตอนโมเดลโหลดเสีย) |
| B15 | `int(steps)` ตัดเศษของ pitch shift | `backend/routers/audio_ops.py:170` | FREE ส่ง `steps=2.99` ผ่านเช็ค (int=2) แต่ shift จริง 2.99 |
| B16 | CNN load ด้วย `strict=False` + filter shape ที่ไม่ตรงทิ้งเงียบๆ | `backend/auto_eq_inference.py:175` | checkpoint ผิดรูปแบบจะได้โมเดลน้ำหนักสุ่มโดยไม่เตือน; `del model` ที่ `:445` ไม่ได้คืน RAM จริงเพราะ `lru_cache` ถือ reference อยู่ |

---

## ⚛️ Frontend / API Routes (Next.js)

### 🔴 Critical

| # | ปัญหา | ตำแหน่ง |
|---|-------|---------|
| F1 | ~~Webhook Omise ปลอมแปลงได้ (fail-open)~~ ✅ แก้แล้ว TU-4 | ~~`app/api/webhooks/omise/route.ts:59-67`~~ | verify `req.text()` raw body + `timingSafeEqual`; ไม่มี secret → 503; ไม่มี/ผิด signature → 403 (fail-closed) |
| F2 | ~~Cancel/Expire แล้ว tier ไม่ถูกลด~~ ✅ แก้แล้ว TU-5 | ~~`quota/consume/route.ts:25-26`~~ | เพิ่ม `lib/subscription.ts#getEffectiveTier()` — สิทธิ์อิง status ACTIVE + periodEnd ยังไม่หมด; ใช้ร่วมกับ `account/route.ts`; แก้เพิ่ม: quota ใช้ค่า monthlyQuota ที่คำนวณใหม่แทนค่า stale ใน record |
| F3 | ~~Checkout ซ้ำไม่ทำลาย schedule เก่า~~ ✅ แก้แล้ว TU-5 | ~~`checkout/route.ts:54-66`~~ | destroy `omiseScheduleId` เดิมก่อนสร้าง schedule ใหม่ (catch + เดินหน้าต่อถ้า schedule หายแล้ว) |

### 🟠 High

| # | ปัญหา | ตำแหน่ง | รายละเอียด |
|---|-------|---------|-----------|
| F4 | ~~TOCTOU race ในการหัก quota~~ ✅ แก้แล้ว TU-8 | ~~`app/api/quota/consume/route.ts:43-58`~~ | conditional atomic `updateMany` (ตรวจ `usedCount < max` ใน where) + unique `(userId, periodStart)` + migration `20260731120000_add_usage_quota_unique` + `upsert` กัน period ซ้ำ |
| F5 | ~~Schedule charge ไม่ส่ง metadata~~ ✅ แก้แล้ว TU-4 | ~~`app/api/subscription/checkout/route.ts:59`~~ | เพิ่ม `charge.metadata: { userId, tier }` → webhook schedule.process ผูก user ได้ |
| F6 | ~~ให้ tier ACTIVE ก่อน charge สำเร็จครั้งแรก~~ ✅ แก้แล้ว TU-5 | ~~`checkout/route.ts:49-66`~~ | CREDIT_CARD สร้าง subscription เป็น `PENDING` — ACTIVE ต่อเมื่อ webhook charge สำเร็จ |
| F7 | ~~`cardToken` ใหม่ถูกเพิกเฉยถ้ามี customer อยู่แล้ว~~ ✅ แก้แล้ว TU-5 | ~~`checkout/route.ts:36-47`~~ | `omise.customers.update(customerId, { card: cardToken })` attach การ์ดใหม่เข้ากับ customer เดิม |
| F8 | ~~บันทึก PaymentRecord "successful" ก่อนเช็ค status~~ ✅ แก้แล้ว TU-4 | ~~`app/api/webhooks/omise/route.ts:75-77`~~ | บันทึกด้วยสถานะจริง; `failed`/`pending` ไม่ activate subscription |
| F9 | ~~User enumeration + ไม่มี rate limit~~ ✅ แก้แล้ว TU-9 | ~~`lib/auth.ts:33-40`~~ | error เดียวกัน "Invalid email or password" + เทียบ bcrypt กับ dummy hash เสมอ (กัน timing) + `middleware.ts` rate limit 10 ครั้ง/นาที/IP ที่ login/register/เปลี่ยนรหัสผ่าน (in-memory per-process) |
| F10 | ~~`container.innerHTML = ""` ลบ DOM ที่ React จัดการอยู่~~ ✅ แก้แล้ว TU-6 | ~~`app/components/AdvancedMultiTrackPlayer.tsx:104`~~ | แยก WaveSurfer container เป็น ref เดี่ยว + gradient overlay เป็น sibling (React ดูแล) → ไม่มี crash ตอน unmount; กัน double destroy; ปุ่ม Play disabled จนทุก stem ready; polish reload เฉพาะ vocals; polish error แสดง feedback |
| F11 | ~~หักโควตาก่อนประมวลผล ไม่คืนเมื่อ fail~~ ✅ แก้แล้ว TU-1+TU-7 | ~~`UploadBox.tsx:358-367`~~ | backend: validate ก่อนนับ (TU-1) • frontend: `quotaCharged` flag + `/api/quota/refund` (decrement แบบ `gt: 0`) คืนโควตาเมื่อ fail/cancel |
| F12 | ~~Export ข้าม format หักโควตาซ้ำ + ประมวลผลใหม่ทั้งไฟล์~~ ✅ แก้แล้ว TU-7+ต่อยอด | ~~`UploadBox.tsx:534-554`~~ | ใหม่ `/convert-format` endpoint: แปลงไฟล์ที่ประมวลผลแล้ว (wav↔mp3) โดย**ไม่ต้องประมวลผลใหม่/ไม่หักโควตา**; `handleSingleExport` ใช้ convert + revoke blob เก่า + auto-download |

### 🟡 Medium

| # | ปัญหา | ตำแหน่ง |
|---|-------|---------|
| M1 | ~~WaveSurfer refs ไม่ set null หลัง destroy + double destroy~~ ✅ แก้แล้ว TU-6 | ~~`AdvancedMultiTrackPlayer.tsx:106, 158-163`~~ |
| M2 | ~~Toggle Vocal Polish rebuild player ทั้ง 4 ตัว~~ ✅ แก้แล้ว TU-6 (reload เฉพาะ vocals, ใช้ ref กัน stale) | ~~`AdvancedMultiTrackPlayer.tsx:164`~~ |
| M3 | ~~สถานะ Solo หายหลัง rebuild~~ ✅ แก้แล้ว TU-6 (syncVolume อ่าน state ผ่าน ref) | ~~`AdvancedMultiTrackPlayer.tsx:133`~~ |
| M4 | ~~กด Play ก่อน ready ครบ → stems หลุด sync ถาวร~~ ✅ แก้แล้ว TU-6 (disabled จนกว่า readyMap ครบ) | ~~`AdvancedMultiTrackPlayer.tsx:204-212`~~ |
| M5 | ~~`audioprocess` ×4 ตัว → setState ~240 ครั้ง/วินาที~~ ✅ แก้แล้ว (throttle 100ms → ~10 ครั้ง/วินาที) | `AdvancedMultiTrackPlayer.tsx` |
| M6 | ~~Vocal Polish ล้มเหลวเงียบ~~ ✅ แก้แล้ว TU-6 (error feedback + AbortController) | ~~`AdvancedMultiTrackPlayer.tsx:255-279`~~ |
| M7 | ~~`URL.createObjectURL` ไม่เคย revoke~~ ✅ แก้แล้ว TU-7 | ~~`UploadBox.tsx:404, 436, 456`~~ | `downloadUrlRef` + `clearDownloadUrl()` — revoke ตอนเลือกไฟล์ใหม่/upload ใหม่/unmount |
| M8 | ~~เลือกไฟล์ใหม่แล้วผลลัพธ์เก่ายังค้างบนหน้าจอ~~ ✅ แก้แล้ว TU-7 | ~~`UploadBox.tsx:255-275`~~ | `handleFileSelect` ล้าง downloadUrl/fileId/zipUrl/analysis/processingTime ครบ |
| M9 | ~~ลิงก์ Karaoke/download เป็น cross-origin~~ ✅ แก้แล้ว (Group B) | `UploadBox.tsx` + `history/page.tsx` + ใหม่ `lib/download.ts#downloadViaBlob` | fetch → blob → objectURL → download; ไฟล์หมด TTL → toast แจ้งแทน navigate 404 |
| M10 | ~~Play/Download ใน History ใช้ไฟล์หมดอายุเงียบๆ~~ ✅ แก้แล้ว (Group B) | `history/page.tsx` | `audio.onerror` + play catch → toast "ไฟล์หมดอายุแล้ว"; delete fail → toast |
| M11 | ~~`SingleExportModal` จำ format ครั้งแรกไว้ (stale state)~~ ✅ แก้แล้ว (Group A) | `UploadBox.tsx` — ใช้ `key={exportFormat-isOpen}` บังคับ remount |
| M12 | ~~พิมพ์ค่า Pitch ติดลบไม่ได้~~ ✅ แก้แล้ว (Group A) | `PitchShiftSettings.tsx` — raw string state + แปลงเมื่อพิมพ์สมบูรณ์ (รับ "-"/"." ระหว่างทาง) |
| M13 | ~~`<style>` global ใน WaveformPlayer รั่วไหลไปทั้งแอป~~ ✅ แก้แล้ว (Group A) | `WaveformPlayer.tsx` — scope selector ด้วย `.waveform-player-volume` prefix |
| M14 | ~~Trim ไม่ validate ฝั่ง client~~ ✅ แก้แล้ว (Group A + B8 backend) | `UploadBox.tsx` — block ก่อน submit ถ้า start<0 / end<=start |
| M15 | ~~`app/components/studio/` 4 ไฟล์เป็น dead code~~ ✅ ลบแล้ว TU-10 | ~~`AudioIngestionBox.tsx`, `AudioToolSelector.tsx`, `AudioResultView.tsx`, `AudioDropzone.tsx`~~ | ลบทั้ง 4 ไฟล์ + `backend/services/audio_workspace.py` (dead code ซ้ำทั้งไฟล์) + `test_audio_workspace.py` — ประหยัด ~900 บรรทัด |
| M16 | ~~Register ไม่ validate password/email; email case-sensitive~~ ✅ แก้แล้ว TU-9 | ~~`app/api/auth/register/route.ts:9-18`~~ | normalize email (lowercase/trim), validate format + password ≥6, name ต้องเป็น string, duplicate → 409, P2002 race → 409 |
| M17 | ~~รั่ว error message ภายในผ่าน HTTP 500~~ ✅ แก้แล้ว (Group C) | `checkout/route.ts`, `webhooks/omise/route.ts` (register เป็น generic อยู่แล้วตั้งแต่ TU-9) | log ฝั่ง server + ตอบข้อความ generic |
| M18 | ไม่ validate `tier` ที่ checkout — จ่ายเงินแล้วได้ FREE quota | `checkout/route.ts:14-15` |
| M19 | ~~OAuth-only user ลบ account/cancel ได้โดยไม่ยืนยันตัวตน~~ ✅ แก้แล้ว TU-9 | ~~`account/route.ts:75-83`, `cancel/route.ts:32-40`~~ | ต้องกรอก `confirmEmail` ตรงกับอีเมลบัญชี (case-insensitive) ก่อนลบ/ยกเลิก — re-auth เต็มรูปแบบเก็บไว้เป็นงานต่อยอด |
| M20 | ~~Unlink provider ตัวสุดท้ายได้ → lockout ถาวร~~ ✅ แก้แล้ว TU-9 | ~~`account/providers/route.ts:39-59`~~ | ปฏิเสธถ้าเหลือ provider แค่ 1 และไม่มี password สำรอง |
| M21 | ~~เปลี่ยน email ไม่ยืนยัน + TOCTOU + ไม่มี try/catch → 500 ดิบ~~ ✅ แก้แล้ว | `account/profile/route.ts` | เปลี่ยน email ต้องยืนยันตัวตน (password / confirmEmail สำหรับ OAuth) + normalize + P2002 → 409; email verification flow เต็มรูปแบบ (ส่งลิงก์ยืนยัน) ยังเป็นงานต่อยอด |

### 🟠 Low (สรุปย่อ)

- **L1** ~~`lib/auth.ts:108-119` — JWT `token.tier` stale หลัง upgrade~~ ✅ แก้แล้ว (Group D) — session callback อ่าน tier ล่าสุดจาก DB (fallback token เมื่อ DB fail)
- **L2** ~~`lib/auth.ts:79-107` — findUnique-then-create race ตอน signIn~~ ✅ แก้แล้ว (Group D) — ใช้ `subscription.upsert` + `usageQuota.upsert` (unique key จาก TU-8)
- **L3** `account/route.ts:29-32` — ~~fallback quota ไม่เช็ค periodEnd~~ ✅ แก้แล้ว TU-5 (ใช้ getEffectiveTier)
- **L4** ~~`account/route.ts:48` — ส่ง `omiseScheduleId` (internal ID) ถึง client~~ ✅ แก้แล้ว (strip ออกจาก response)
- **L5** ~~`lib/omise.ts:14-25` — monkey-patch `https.request` ระดับ global ไม่ idempotent~~ ✅ แก้แล้ว (Group D) — marker กัน patch ซ้อน (HMR-safe) + รองรับ URL object
- **L6** ~~`history/[id]/route.ts:17-25` — DELETE race~~ ✅ แก้แล้ว — `deleteMany({ id, userId })` atomic
- **L7** ~~`lib/auth.ts:52-53` — Facebook scope ไม่ขอ email~~ ✅ แก้แล้ว (Group D) — scope `public_profile,email`
- **L8** ~~`webhooks/omise/route.ts:17-37` — renew ไม่ update `currentPeriodStart`; UsageQuota แถวใหม่ทุกครั้ง~~ ✅ แก้แล้ว (Group D) — `deleteMany` รอบเก่า + create รอบใหม่ (เหลือแถวเดียว)
- **L9** ~~`subscription/history` + `history` — ไม่มี pagination~~ ✅ แก้แล้ว (Group D) — `take: 50` + cursor-based pagination
- **L10** ~~`history/route.ts:26-27` — POST ไม่ guard JSON parse~~ ✅ แก้แล้ว (400 สำหรับ non-JSON, validate stems/length) | `UploadBox.tsx:203-207` — action naming 3 ระบบ ยังค้าง (ย้ายไป Phase refactor)

---

# ส่วนที่ 2: Clean Code / DRY / SOLID

## Backend

| # | ปัญหา | ตำแหน่ง |
|---|-------|---------|
| C1 | ~~`audio_workspace.py` ซ้ำซ้อนทั้งไฟล์~~ ✅ ลบแล้ว TU-10 | ~~`backend/services/audio_workspace.py`~~ |
| C2 | ~~Logic mixdown ซ้ำ~~ ✅ แก้แล้ว — `mixdown_stems()` ใน `stems.py` ใช้ร่วม karaoke + export mix (จัดการ mono/stereo ปนกัน + normalize) | `stems.py` |
| C3 | ~~Logic zip ซ้ำ~~ ✅ แก้แล้ว — `create_zip_archive()` ใช้ร่วม `/separate` + export (และ `/separate` เปลี่ยนเป็น `to_thread` แก้ B10) | `stems.py` |
| C4 | ~~Tier/quota mapping กระจาย 4 จุด~~ ✅ แก้แล้ว — `lib/subscription.ts#TIER_MONTHLY_QUOTA` + `getMonthlyQuota()` | `lib/subscription.ts` |
| C5 | ~~Magic numbers~~ ✅ แก้แล้ว — ราคา → `TIER_PRICES`/`getTierPrice`; `30 วัน` → `PERIOD_MS` ใน `lib/config.ts`; dead `_cleanup_partial_checkpoints` (และ magic `108000000`) ถูกลบ | `lib/config.ts`, `process_audio.py` |
| C6 | ~~Error handling ซ้ำทุกรูท (try/except → 500)~~ ✅ แก้แล้ว | `backend/main.py#unhandled_exception_handler` (global) + ลบ generic `except Exception` ออกจาก 6 endpoints (`/separate`, `/karaoke`, apply-eq-ai, apply-compressor, pitch-shift, analyze) — เก็บ handlers เฉพาะ (HTTPException/ValueError/AutoEQModelLoadError) + `finally` cleanup; error detail ไม่รั่วถึง client (M17) |

## Frontend

| # | ปัญหา | ตำแหน่ง |
|---|-------|---------|
| C7 | ~~**`UploadBox.tsx` คือ God Component~~ ✅ แก้แล้ว | `lib/hooks/useAudioProcessor.ts` (ใหม่ 542 บรรทัด) + `UploadBox.tsx` ลดจาก ~1077 → 662 บรรทัด | แยก business logic (upload/process/export/quota/history/trim validation) ออกจาก UI — ทดสอบผ่าน UI tests เดิมครบ |
| C8 | ~~Action naming 3 ระบบ~~ ✅ แก้แล้ว — `AUDIO_ACTIONS` + `ACTION_TO_BACKEND` ใน `lib/config.ts`; UploadBox ใช้ `AudioAction` type + `saveHistory` ใช้ map | `lib/config.ts` |
| C9 | ~~Auth guard ซ้ำ 10+ routes~~ ✅ แก้แล้ว — `requireSession()` ใน `lib/auth.ts` ใช้ครบ 13 routes | `lib/auth.ts` |
| C10 | ~~Password-confirm block ซ้ำ~~ ✅ แก้แล้ว — `verifyAccountAuth()` ใน `lib/auth.ts` ใช้ร่วม account DELETE + cancel | `lib/auth.ts` |
| C11 | ~~Skeleton UI ซ้ำ~~ ✅ หมดไปเมื่อลบ `AudioResultView.tsx` (TU-10) | — |
| C12 | ~~Prop drilling (17 props)~~ ✅ แก้แล้ว — `CompressorParams` object + `onChange(patch)` + field config array (DRY markup 8 ชุด) | `CompressorSettings.tsx` |
| C13 | ~~augment JWT ผิด module~~ ✅ แก้แล้ว — ย้ายไป `declare module "next-auth/jwt"` + fields เป็น optional | `types/next-auth.d.ts` |
| C14 | ~~Magic string action กระจาย~~ ✅ แก้แล้ว — `AudioAction` union + `ACTION_TO_BACKEND` | `lib/config.ts` |
| C15 | ~~`PERIOD_MS` ซ้ำ 4 ที่~~ ✅ แก้แล้ว — รวมใน `lib/config.ts` ใช้ครบ 4 จุด | `lib/config.ts` |
| C16 | ~~Checkout ผสม 2 flow + webhook if-chain~~ ✅ แก้แล้ว — `createCardSubscription()` แยก flow; webhook ใช้ handler map | `checkout/route.ts`, `webhooks/omise/route.ts` |

---

# ส่วนที่ 3: Test Cases ที่ควรเพิ่ม

> สถานะปัจจุบัน: `tests/*.test.ts` ครอบคลุมแค่ 401/happy path; `backend/tests/` มีแค่ `test_auth_guard.py` — ยังขาด regression tests ของ bug ทั้งหมดด้านล่าง

## Backend (pytest) — เรียงตามความสำคัญ

| # | Test Case | ครอบคลุม Bug |
|---|-----------|--------------|
| T1 | Guest ส่ง `X-User-Tier: PRO/BASIC` → ต้องถูกปฏิเสธหรือถูกนับโควตาเหมือน FREE | B1 |
| T2 | `/apply-compressor` ด้วย `genre=../../x` และ genre แปลก → 400; `strength` ผิด → 400 | B2 |
| T3 | Export stems แบบ MP3 เสร็จแล้ว → `vocals.wav` ต้นฉบับต้องยังอยู่ | B3 |
| T4 | `polish_vocal_file` / `apply_lufs_mastering` กับไฟล์ stereo → output ต่างจาก input จริงและไม่มี NaN; ไฟล์ silence → ไม่มี NaN; mono ทำงานปกติ | B4, B5 |
| T5 | `save_upload`: trim ติดลบ / start>end / เกินความยาว → 400; ไฟล์ >100MB → 400; ไม่ใช่ .wav → 400 | B8 |
| T6 | ไฟล์โควตา JSON เสีย → fail-closed ไม่ใช่รีเซ็ตโควตา (regression B7) | B7 |
| T7 | อัปโหลดไฟล์ผิดนามสกุล → โควตาไม่ถูกหัก; processing 500 → โควตาไม่ถูกหัก (หรือคืน) | B6 |
| T8 | `convert_to_mp3` ล้มเหลว (mock pydub/ffmpeg) → ต้อง raise/ส่งสัญญาณ ไม่ใช่คืน wav path เงียบๆ | B9 |
| T9 | Karaoke mixdown ด้วย stems ขนาด/ช่องเสียงต่างกัน → ไม่ 500 | B12 |
| T10 | `/analyze` ไฟล์ 0 sample / เสียหาย → 400 ไม่ใช่ 500 | B13 |
| T11 | `process_export` ไม่เลือก stem → 400; export mix → LUFS ใกล้เคียง target | stems.py |
| T12 | AutoEQ: model_id ไม่รองรับ → 400; โมเดลไฟล์หาย → 503 AUTO_EQ_MODEL_UNAVAILABLE | auto_eq_inference |

## Frontend (Jest) — เรียงตามความสำคัญ

| # | Test Case | ครอบคลุม Bug |
|---|-----------|--------------|
| T13 | Webhook: ไม่ส่ง signature + มี secret ตั้งอยู่ → 403; signature ผิด → 403 | F1 |
| T14 | Webhook: `charge.complete` status=failed → ไม่ activate subscription + PaymentRecord เป็น "failed" | F8 |
| T15 | Webhook: event ไม่รู้จัก → 200 แต่ไม่แตะ DB; metadata ไม่มี userId → ไม่ crash | F1 |
| T16 | `quota/consume`: 2 request พร้อมกันเหลือโควตา 1 → สำเร็จแค่ 1 (integration กับ SQLite จริง) | F4 |
| T17 | ผู้ใช้ status=CANCELED/EXPIRED แต่ tier=PRO → ได้ quota แบบ FREE; periodEnd ผ่าน → reset | F2 |
| T18 | Checkout: tier ไม่ใช่ BASIC/PRO → 400; มี schedule เก่า → destroy ก่อนสร้างใหม่; มี customer + cardToken ใหม่ → attach เข้า customer เดิม | F3, F7 |
| T19 | Register: password <6 → 400; email ต่าง case → 409; name เป็น object → 400 ไม่ใช่ 500 | M16 |
| T20 | Login email ไม่มีอยู่ vs password ผิด → error message เดียวกัน | F9 |
| T21 | UploadBox: ไฟล์ผิด/ใหญ่เกิน → reject; เลือกไฟล์ใหม่ → ผลเก่าถูกล้าง; quota 403 → หยุดก่อนส่งไฟล์; unmount ระหว่าง loading → ไม่มี setState | F11, M8 |
| T22 | UploadBox: `handleSingleExport` เปลี่ยน format → ไม่เรียก quota ซ้ำ; trim start > end → block ก่อนส่ง | F12, M14 |
| T23 | Player: ไม่มีการเขียน innerHTML ลง container ที่ React render; Play ก่อน ready → ไม่หลุด sync; solo + rebuild → volume ถูกต้อง; polish API !ok → มี feedback; unmount ระหว่าง fetch → ไม่มี setState | F10, M1-M4, M6 |
| T24 | `PitchShiftSettings`: พิมพ์ "-2" → ค่าเป็น -2 ไม่ใช่ 0; clamp ตาม tier | M12 |
| T25 | `SingleExportModal`: เปิดใหม่หลัง format เปลี่ยน → radio ตรงกับค่าปัจจุบัน | M11 |
| T26 | History API: POST body ไม่ใช่ JSON → 400; DELETE id ของคนอื่น → 403; DELETE id ไม่มี → 404 | L11, L12 |

---

## ข้อแนะนำลำดับการแก้ไข (Priority Order)

### Phase 1 — ปิดช่องโหว่ Critical (กระทบรายได้/ความปลอดภัย)
1. **F1** — Webhook: ใช้ `await req.text()` + ตรวจ signature แบบ fail-closed; ตั้ง `OMISE_WEBHOOK_SECRET` (หรือยืนยัน event id ย้อนกลับที่ Omise API)
2. **F2** — Downgrade tier เมื่อ status ≠ ACTIVE หรือ periodEnd ผ่านไป (ที่ `quota/consume` และเพิ่ม cron/job)
3. **B1** — Backend ไม่ trust client header: derive tier จาก user_id ฝั่ง server (หรือใช้ signed token); guest ต้องถูกนับโควตาเสมอ
4. **F3** — Destroy schedule เก่าก่อนสร้างใหม่ (wrap ใน transaction)
5. **B3** — `convert_to_mp3` ต้อง copy แทน move สำหรับไฟล์ stem ต้นฉบับ (หรือแปลงเป็น temp แล้วค่อย rename)
6. **B2** — เพิ่ม pattern validation ให้ `genre` + whitelist ชื่อไฟล์ output
7. **B4** — Transpose ก่อนส่งเข้า pedalboard: `board(data.T, sr).T` (รองรับทั้ง mono/stereo)
8. **B5** — เช็ค `np.isinf(current_lufs)` / `np.isnan` ก่อนคำนวณ gain

### Phase 2 — High (ความเสถียร)
- F4 (atomic quota update), F5-F9, B6 (quota refund/validate ก่อนหัก), B7 (atomic JSON write), B8 (trim validation), F10 (แยก ref สำหรับ WaveSurfer container), F11, F12

### Phase 3 — Medium/Low + Code Quality
- M1-M21, L1-L13, C1-C16 (เริ่มจาก: ลบ dead code `studio/*` + `audio_workspace.py`, แยก `useAudioProcessor`, รวม constants กลาง)

---

*รายงานนี้สร้างจากผลการตรวจสอบโค้ดฉบับเต็ม — ทุกข้ออ้างอิง file:line จริงใน repository*
