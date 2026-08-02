# Production User Management & Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. This document is the master plan; each workstream should be split into an independently testable implementation plan before coding. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ยกระดับระบบ User Management ของ HarmoniQ จากระบบ auth ระดับ prototype ให้รองรับ production โดยมี audit trail, structured logging, abuse prevention, account recovery, session control, privacy lifecycle, admin controls และการตรวจสอบสิทธิ์ระหว่าง Next.js กับ FastAPI อย่างถูกต้อง

**Architecture:** ให้ Next.js/NextAuth เป็นแหล่งยืนยันตัวตนหลักและเป็นผู้คำนวณสิทธิ์ subscription จากฐานข้อมูล ส่วน FastAPI รับเฉพาะ short-lived signed service token หรือถูกเรียกผ่าน Next.js BFF แทนการเชื่อ `X-User-ID` และ `X-User-Tier` จาก client โดยตรง แยก security/business audit log ที่ต้องเก็บตาม retention policy ออกจาก operational log ที่ส่งไปยัง log collector และใช้ Redis สำหรับ rate limit/short-lived state เมื่อ deploy มากกว่าหนึ่ง process

**Tech Stack:** Next.js 15 App Router, NextAuth v4, Prisma 6, PostgreSQL สำหรับ production, SQLite สำหรับ local development, Redis/Upstash Redis, FastAPI/Python 3.10, PyJWT, Jest, pytest, email provider ผ่าน adapter interface, structured JSON logging

## Global Constraints

- Backend ต้องเข้ากันได้กับ Python 3.10 ห้ามใช้ syntax หรือ typing feature ของ Python 3.11+
- ทุก database operation ต้องผ่าน Prisma ฝั่ง Next.js เว้นแต่มีการอนุมัติให้ backend อ่านฐานข้อมูลเดียวกันในแผนย่อย
- ห้าม log password, password hash, OAuth access/refresh token, session token, JWT, CSRF token, API key, Omise secret, card data หรือเนื้อหาไฟล์เสียง
- IP, email, user-agent และชื่อไฟล์เป็นข้อมูลส่วนบุคคล ต้อง hash, mask, truncate หรือเก็บเท่าที่จำเป็นและมี retention ที่ชัดเจน
- Audit log ต้องไม่ถูกลบตาม cascade เมื่อผู้ใช้ลบบัญชี ต้อง anonymize หรือทำให้ `userId` เป็น `null` ตาม retention/legal policy
- ใช้ `NEXTAUTH_SECRET` เฉพาะสำหรับ NextAuth และใช้ secret/key แยกสำหรับ token ที่ Next.js ออกให้ FastAPI
- เปลี่ยนค่า tier, quota, subscription status และสิทธิ์ admin ได้จาก server-side source of truth เท่านั้น
- ทุก download/playback/export ของ audio artifact ต้องผ่าน owner check หรือ short-lived file grant ที่ผูกกับ owner, file ID, action และ expiry; ห้ามถือว่า file ID ที่เดายากเป็น authorization
- quota ต้องใช้ server-side reservation/operation ID ที่ unique และ refund ได้เฉพาะ reservation เดิม; dependency ล้มเหลวต้อง fail-closed ก่อนเริ่มงาน authenticated
- ห้ามใช้ production seed ที่พิมพ์หรือกำหนด password ลง log; admin bootstrap ต้องใช้ secret manager/one-time invite และ step-up authentication
- การแก้ไข state ผ่าน cookie-authenticated API ต้องมี SameSite cookie และ origin/CSRF protection; webhook และ service-to-service API ต้องใช้ signature/token ของตัวเอง
- ทุก feature ต้องมี Jest/pytest test ที่ครอบคลุม success, unauthorized, abuse path, replay/race condition และ failure ของ dependency ที่เกี่ยวข้อง
- การแก้ไข UI ต้องอ่าน `DESIGN.md` ก่อน และต้องรองรับ desktop/mobile ตาม design system เดิม
- ห้ามรัน `npm run build` คู่กับ dev server; ใช้ `npm run type-check`, `npm run lint`, `npx jest` และ `pytest`/`python -m unittest discover -s backend/tests` ตามงานที่เกี่ยวข้อง

---

## 1. Current Baseline

### สิ่งที่มีอยู่แล้ว

| พื้นที่ | สถานะปัจจุบัน | ไฟล์หลัก |
|---|---|---|
| Authentication | NextAuth v4, Credentials + Google + Facebook + LINE | `lib/auth.ts`, `app/api/auth/[...nextauth]/route.ts` |
| Session | JWT strategy; มี Prisma `Session` model แต่ไม่ได้ใช้เป็น source of truth | `lib/auth.ts`, `prisma/schema.prisma` |
| User data | `User`, `Account`, `Session`, `VerificationToken`, `Subscription`, `UsageQuota`, `PaymentRecord`, `ProjectRecord` | `prisma/schema.prisma` |
| Account settings | profile, password, providers, preferences, export/delete มี route อยู่แล้วบางส่วน | `app/api/account/` |
| Rate limit | in-memory `Map`, 10 requests/minute/IP สำหรับ auth paths | `middleware.ts`, `lib/rate-limit.ts` |
| Backend auth | รับ `X-User-ID`/`X-User-Tier`; guest quota ใช้ไฟล์ JSON ตาม IP | `backend/utils/auth_guard.py` |
| Backend logging | Python standard logging และ global exception handler | `backend/main.py` |
| File lifecycle | ไฟล์เสียงชั่วคราวมี TTL cleanup; `ProjectRecord` เก็บ processing history | `backend/cleanup_task.py`, `prisma/schema.prisma` |

### Gaps ที่ต้องปิดก่อนเรียกว่า production-ready

1. ไม่มี audit trail สำหรับ login, password change, provider change, subscription change, account deletion และ admin action
2. Operational log ยังเป็นข้อความจาก `console.*`/`logging.basicConfig` ไม่มี request ID, event name, severity policy และ redaction ที่สม่ำเสมอ
3. Rate limit หายเมื่อ process restart และไม่แชร์ระหว่าง instance
4. Backend เชื่อ header ที่ client ปลอมได้ และไม่มีการตรวจ signature ของ identity
5. JWT session ยกเลิกราย device ไม่ได้
6. `emailVerified` มีใน schema แต่ยังไม่มี verification flow ที่ใช้จริง
7. ยังไม่มี password reset แบบ one-time token ที่ hash ใน database
8. Export/delete ยังต้องเพิ่ม retention, async job, anonymization และการจัดการไฟล์ backend
9. ยังไม่มี RBAC/admin audit trail
10. `emailNotifications` ยังไม่ใช่ notification delivery system หรือ consent record
11. Custom mutation routes ต้องมีมาตรฐาน CSRF/origin/security headers ที่ชัดเจน

---

## 2. Production Target Architecture

### 2.1 Component diagram

```text
Browser
  |
  | SameSite HttpOnly session cookie
  v
Next.js Web/API (BFF + NextAuth)
  |-- PostgreSQL via Prisma: users, sessions, billing, audit, tokens
  |-- Redis: rate limits, short-lived locks, notification/outbox state
  |-- Email provider adapter: verification, reset, security notices
  |
  | short-lived signed internal token, or private BFF proxy
  v
FastAPI audio service
  |-- verifies issuer/audience/signature/expiry/scope
  |-- enforces processing limits and guest quota
  |-- owns temporary audio cleanup
```

### 2.2 Architecture decisions

| ประเด็น | ค่าเริ่มต้นที่แนะนำ | เหตุผล |
|---|---|---|
| Production database | PostgreSQL | SQLite เหมาะกับ local/single-process แต่ไม่เหมาะกับ concurrent audit writes, worker และหลาย instance |
| Local database | SQLite ผ่าน Prisma | รักษา developer experience เดิม และใช้ schema เดียวกันเท่าที่ Prisma รองรับ |
| Distributed state | Redis/Upstash Redis | ใช้กับ rate limit, lockout counter, idempotency และ queue coordination |
| Auth source of truth | NextAuth + User/Subscription ใน Next.js | ระบบปัจจุบันอยู่ฝั่งนี้และ backend ไม่มี business database |
| Session strategy | Database session เมื่อเปิด device management | JWT ล้วน revoke ราย device ไม่ได้โดยไม่เพิ่ม state check ทุก request |
| Backend trust | BFF/private network เป็นค่าเริ่มต้น; signed token เป็น fallback สำหรับ direct upload | ไม่เปิด secret ใน browser และไม่ trust custom headers |
| Token signing | HMAC secret แยกในระบบเดียว; asymmetric key/JWKS เมื่อมีหลาย service | ลด coupling กับ `NEXTAUTH_SECRET` และรองรับ rotation |
| Audit storage | Prisma `AuditLog` ใน PostgreSQL พร้อม retention job | ใช้ query ตรวจสอบ account/security/admin activity ได้ |
| Operational logs | JSON stdout ไปยัง collector/Sentry/Datadog/Loki ตาม deployment | ไม่ทำให้ database โตจากทุก request และรองรับ search/alert |
| Email | provider adapter โดย default ใช้ Resend หรือ SES | เปลี่ยน provider ได้โดยไม่ผูก route เข้ากับ SDK โดยตรง |

### 2.3 Boundary rules

- Client ห้ามส่ง tier/user ID เพื่อขอสิทธิ์; ถ้ามี header เดิมให้ถือเป็น untrusted และลบออกจาก client เมื่อ migration เสร็จ
- FastAPI ที่รับ `Authorization: Bearer ...` ต้อง reject token invalid อย่างชัดเจน ไม่ downgrade เป็น guest เมื่อมี token ที่ malformed/expired
- Guest ที่ไม่มี Authorization จึงเข้ากระบวนการ guest quota ได้เท่านั้น
- Next.js ต้อง consume quota จาก database ก่อนส่งงาน authenticated ไป backend และ refund แบบ idempotent เมื่อ backend ล้มเหลว
- Backend token ต้องมี `sub`, `tier`, `scope`, `iss`, `aud`, `iat`, `exp`, `jti`, `kid` เท่าที่จำเป็น และมีอายุไม่เกิน 5 นาที
- โหมด signed token ที่ browser เรียกตรงต้องมี single-use `jti`/idempotency binding ใน Redis; ถ้าทำ replay protection ไม่ได้ให้ใช้ BFF-only
- production default ตั้ง `BACKEND_DIRECT_TOKEN_ENABLED=false`; FastAPI รับเฉพาะ BFF/private service call และ signed token ที่ออกให้ browserตรงเป็น opt-in deployment profile ที่ต้องผ่าน replay/IDOR tests แยก
- ทุก service ต้องส่ง `X-Request-ID` เดิมตลอด request chain; ถ้า client ส่งค่ามาให้ validate รูปแบบและสร้างใหม่เมื่อไม่ปลอดภัย

### 2.4 Concrete deployment choices for the first production slice

- Next.js middleware/Edge-compatible Redis: `@upstash/redis`; ถ้า target เป็น long-running Node server ให้เลือก `ioredis` แทนทั้งชุด ไม่ติดตั้งสองแบบโดยไม่มีเหตุผล
- FastAPI token verification: `PyJWT` major 2 พร้อม algorithm allowlist; ไม่ใช้ NextAuth JWE decoder ใน Python และไม่แชร์ `NEXTAUTH_SECRET`
- Email: ใช้ `resend` adapter เป็น default implementation หลัง verify domain; interface ต้องรองรับ SES/SMTP ในอนาคต
- OAuth token encryption: ใช้ envelope encryption ผ่าน KMS หรือ `APP_ENCRYPTION_KEY` แบบ versioned; ไม่ใช้ PrismaAdapter ตรง ๆ กับ production token fields โดยไม่มี wrapper
- Object storage: single-instance pilot ใช้ local disk + TTL เดิมได้; multi-instance production ต้องใช้ S3-compatible private bucket สำหรับ artifact/export และ signed/private access
- Worker: long-running deployment ใช้ process แยกสำหรับ notification/retention/cleanup; serverless ใช้ scheduler เรียก internal route พร้อม service signature และ lease ใน DB
- BFF upload route ต้องกำหนด `runtime = "nodejs"`, streaming body, 100MB limit, timeout/backpressure และ `maxDuration` ตาม provider ก่อนเปิดใช้งานจริง
- `MAX_CONCURRENT_TASKS` ปัจจุบันเป็น per-process; ถ้ามีหลาย FastAPI process ให้ใช้ Redis semaphore/queue หรือบังคับมี audio worker เดียว ไม่ถือค่า env เดิมว่าเป็น global limit

---

## 3. Data Classification & Logging Policy

### 3.1 Data classes

| Class | ตัวอย่าง | การจัดการ |
|---|---|---|
| Secret | password, password hash, OAuth token, JWT, CSRF token, API key | ห้าม log/export; เก็บใน secret manager หรือ encrypted storage เท่านั้น |
| Direct PII | email, name, IP, user-agent, original filename | เก็บเฉพาะ purpose ที่จำเป็น; mask/hash/truncate; มี retention |
| Financial | Omise customer/charge/schedule ID, amount, payment status | log ด้วย internal ID หรือ last-safe fragment เท่านั้น; ห้ามเก็บ card data |
| Audio content | WAV bytes, stem bytes, waveform, transcript | ห้ามอยู่ใน log/audit; ใช้ `fileId` ที่ไม่เปิดเผยชื่อไฟล์แทน |
| Security metadata | event, outcome, timestamp, request ID, actor/target ID | เก็บใน AuditLog ตาม retention; query เฉพาะ role ที่ได้รับอนุญาต |
| Operational metadata | route, status, duration, service, job ID, error class | structured log; ไม่ใส่ body/header ทั้งก้อน |

### 3.2 สิ่งที่ต้องไม่ log

- request body ของ login, register, password, reset, checkout และ provider callback
- `Authorization`, `Cookie`, `Set-Cookie`, `access_token`, `refresh_token`, `id_token`, `sessionToken`
- password ที่ถูกต้องหรือผิด และ password hash
- email แบบเต็มใน operational log; ใช้ `email_hash` แบบ HMAC หรือ mask เช่น `a***@example.com` เฉพาะ audit ที่จำเป็น
- raw IP ใน log collector; ใช้ `ip_hash` และเก็บ raw IP เฉพาะเมื่อมีข้อกำหนดทางกฎหมาย/incident ที่อนุมัติ
- original filename ถ้าไม่จำเป็น; audit audio ใช้ `file_id` และ action

### 3.3 HMAC hashing

ใช้ HMAC-SHA-256 ด้วย secret คนละตัวสำหรับ identifiers ที่ต้องค้นหาแบบ equality:

```text
HMAC(PII_HASH_SECRET, normalized_email)
HMAC(PII_HASH_SECRET, canonical_ip)
```

ห้ามใช้ plain SHA-256 กับ email/IP เพราะ dictionary attack ทำได้ง่าย และห้ามใช้ `NEXTAUTH_SECRET` เป็น PII hash key เพื่อแยกหน้าที่และรองรับ rotation

---

## 4. Shared Data Model Plan

การเพิ่ม model เหล่านี้ควรทำเป็น migration แยกตาม workstream และต้องทดสอบทั้ง SQLite local กับ PostgreSQL staging ก่อน deploy จริง ค่า enum ใน schema ให้ใช้ `String` ตามรูปแบบปัจจุบันของโปรเจกต์ แล้วรวมค่าที่อนุญาตไว้ใน constants ฝั่ง TypeScript/Python เพื่อให้ SQLite รองรับได้เหมือนเดิม

### 4.1 `User` additions

เพิ่ม field ที่ใช้ควบคุม lifecycle และ abuse state:

```prisma
  role                String    @default("USER")
  status              String    @default("ACTIVE")
  failedLoginCount    Int       @default(0)
  loginFailureWindowStartedAt DateTime?
  lockedUntil         DateTime?
  lastLoginAt         DateTime?
  passwordChangedAt   DateTime?
  deletionRequestedAt DateTime?
  emailVerificationRequiredAt DateTime?

  auditLogs                 AuditLog[]
  emailVerificationTokens  EmailVerificationToken[]
  passwordResetTokens      PasswordResetToken[]
  emailChangeTokens        EmailChangeToken[]
  notificationJobs         NotificationJob[]
  dataRequests             DataRequest[]
  audioArtifacts           AudioArtifact[]
  quotaReservations       QuotaReservation[]
  consents                 UserConsent[]
  adminFactors             AdminFactor[]
```

Allowed values:

- `role`: `USER`, `SUPPORT`, `ADMIN`
- `status`: `ACTIVE`, `PENDING_EMAIL`, `SUSPENDED`, `PENDING_DELETION`, `ANONYMIZED`
- `AudioArtifact.ownerType`: `USER`, `GUEST`, `PURGED`
- `AudioArtifact.status`: `RUNNING`, `COMPLETED`, `FAILED`, `CANCELED`, `UNKNOWN`, `PURGED`
- `QuotaReservation.status`: `RESERVED`, `UNKNOWN`, `RECONCILING`, `CONSUMED`, `REFUNDED`, `EXPIRED`
- `WebhookEvent.status`: `RECEIVED`, `PROCESSING`, `RETRY`, `PROCESSED`, `FAILED`
- `ProcessingJob.phase`: `RESERVED`, `UPLOAD_INTENT_ISSUED`, `UPLOADING`, `PROCESSING`, `SUCCEEDED`, `FAILED`, `UNKNOWN`, `CANCELED`

ไม่เพิ่ม `lastLoginIp` เป็น raw field ใน `User`; ใช้ `AuditLog.ipHash` และ session metadata แทน

`status @default("ACTIVE")` เป็น compatibility default สำหรับ legacy row เท่านั้น; ทุก user creation/import route ต้องระบุ `status`, `emailVerificationRequiredAt` และ `emailVerified` อย่าง explicit และ migration ต้อง backfill ก่อนเปิด policy ไม่ให้ default นี้กลายเป็น verification bypass

### 4.2 `Session` additions

ปัจจุบันมี standard NextAuth fields อยู่แล้ว แต่ JWT strategy ทำให้ model ไม่ถูกใช้จริง เมื่อเปลี่ยนเป็น database session ให้เพิ่ม:

```prisma
  createdAt           DateTime @default(now())
  lastSeenAt          DateTime @default(now())
  lastAuthenticatedAt DateTime @default(now())
  adminMfaVerifiedAt  DateTime?
  adminMfaMethod       String?
  ipHash              String?
  userAgent           String?
  deviceLabel         String?

  @@index([userId])
  @@index([expires])
```

`sessionToken` ต้องไม่ถูกส่งออก API และไม่ถูก log; endpoint จะแสดงเฉพาะ `id`, device label, created/last seen, expiry และ current flag

### 4.3 `AuditLog`

```prisma
model AuditLog {
  id            String    @id @default(cuid())
  userId        String?
  actorType     String
  actorId       String?
  event         String
  outcome       String
  requestId     String?
  ipHash        String?
  userAgent     String?
  targetType    String?
  targetId      String?
  metadata      String?
  piiHashKeyVersion String    @default("v1")
  retentionUntil DateTime
  createdAt     DateTime  @default(now())
  user          User?     @relation(fields: [userId], references: [id], onDelete: SetNull)

  @@index([userId, createdAt])
  @@index([event, createdAt])
  @@index([requestId])
  @@index([retentionUntil])
}
```

`metadata` ต้องเป็น JSON string ที่ผ่าน allowlist และ redaction แล้ว ไม่รับ arbitrary request body จาก route โดยตรง

### 4.4 One-time token models

ใช้ model แยกเพื่อให้ cleanup, index และ retention ชัดเจน:

```prisma
model EmailVerificationToken {
  id             String    @id @default(cuid())
  userId         String
  tokenHash      String    @unique
  expiresAt      DateTime
  usedAt         DateTime?
  requestedIpHash String?
  createdAt      DateTime  @default(now())
  user           User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, expiresAt])
}

model PasswordResetToken {
  id             String    @id @default(cuid())
  userId         String
  tokenHash      String    @unique
  expiresAt      DateTime
  usedAt         DateTime?
  requestedIpHash String?
  createdAt      DateTime  @default(now())
  user           User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, expiresAt])
}

model EmailChangeToken {
  id             String    @id @default(cuid())
  userId         String
  newEmail       String
  tokenHash      String    @unique
  expiresAt      DateTime
  usedAt         DateTime?
  createdAt      DateTime  @default(now())
  user           User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, expiresAt])
}
```

Raw token จะอยู่เฉพาะใน URL/email ชั่วคราว; database เก็บเฉพาะ hash, และทุก token ใช้ได้ครั้งเดียว

### 4.5 Notification and data-request models

```prisma
model NotificationJob {
  id                String    @id @default(cuid())
  userId            String?
  type              String
  channel           String    @default("EMAIL")
  status            String    @default("PENDING")
  dedupeKey         String    @unique
  attemptCount      Int       @default(0)
  scheduledAt       DateTime  @default(now())
  nextAttemptAt     DateTime  @default(now())
  claimedAt         DateTime?
  claimedBy         String?
  claimToken        String?   @unique
  leaseUntil        DateTime?
  payloadCiphertext String?
  recipientCiphertext String?
  providerIdempotencyKey String @unique
  sentAt            DateTime?
  lastError         String?
  providerMessageId String?
  createdAt         DateTime  @default(now())
  user              User?     @relation(fields: [userId], references: [id], onDelete: SetNull)

  @@index([status, nextAttemptAt])
  @@index([status, leaseUntil])
  @@index([userId, createdAt])
}

model DataRequest {
  id          String    @id @default(cuid())
  userId      String
  requestKey  String    @unique
  type        String
  status      String    @default("PENDING")
  storageKey  String?
  downloadTokenHash String? @unique
  downloadedAt DateTime?
  requestedAt DateTime  @default(now())
  completedAt DateTime?
  expiresAt   DateTime?
  errorCode   String?
  exportDeletion ExportDeletion?
  user        User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, requestedAt])
  @@index([status, requestedAt])
}

model ExportDeletion {
  id            String      @id @default(cuid())
  dataRequestId String?     @unique
  storageKey    String
  status        String      @default("PENDING")
  attemptCount  Int         @default(0)
  claimToken    String?     @unique
  leaseUntil    DateTime?
  requestedAt   DateTime    @default(now())
  completedAt   DateTime?
  errorCode     String?
  dataRequest   DataRequest? @relation(fields: [dataRequestId], references: [id], onDelete: SetNull)

  @@index([status, leaseUntil])
  @@index([requestedAt])
}

model AudioArtifact {
  id            String          @id @default(cuid())
  ownerType     String
  userId        String?
  ownerKeyHash  String?
  fileId        String?         @unique
  action        String
  status        String          @default("RUNNING")
  expiresAt     DateTime?
  purgeDeadline DateTime?
  createdAt     DateTime        @default(now())
  completedAt   DateTime?
  cleanedAt     DateTime?
  user          User?           @relation(fields: [userId], references: [id], onDelete: Restrict)
  processingJob ProcessingJob?
  projectRecord ProjectRecord?
  cleanup       ArtifactCleanup?

  @@index([userId, createdAt])
  @@index([ownerType, ownerKeyHash])
  @@index([status, expiresAt])
}

model ProcessingJob {
  id            String        @id @default(cuid())
  operationId   String        @unique
  artifactId    String        @unique
  action        String
  phase         String        @default("RESERVED")
  bodyDigest    String?
  bytesAcceptedAt DateTime?
  leaseUntil    DateTime?
  claimToken    String?       @unique
  lastErrorCode String?
  reconciliationDeadline DateTime?
  reconcileAttemptCount Int    @default(0)
  createdAt     DateTime      @default(now())
  updatedAt     DateTime      @updatedAt
  completedAt   DateTime?
  artifact      AudioArtifact @relation(fields: [artifactId], references: [id], onDelete: Cascade)

  @@index([phase, leaseUntil])
  @@index([artifactId])
}

model ArtifactCleanup {
  id           String        @id @default(cuid())
  artifactId   String        @unique
  status       String        @default("PENDING")
  attemptCount Int           @default(0)
  claimToken   String?       @unique
  leaseUntil   DateTime?
  requestedAt  DateTime      @default(now())
  completedAt  DateTime?
  errorCode    String?
  artifact     AudioArtifact @relation(fields: [artifactId], references: [id], onDelete: Cascade)

  @@index([status, leaseUntil])
  @@index([requestedAt])
}

model QuotaReservation {
  id           String      @id @default(cuid())
  userId       String
  usageQuotaId String
  operationId  String      @unique
  status       String      @default("RESERVED")
  units        Int         @default(1)
  expiresAt    DateTime
  createdAt    DateTime    @default(now())
  consumedAt   DateTime?
  refundedAt   DateTime?
  reconciledAt DateTime?
  reconciliationDeadline DateTime?
  user         User        @relation(fields: [userId], references: [id], onDelete: Restrict)
  usageQuota   UsageQuota  @relation(fields: [usageQuotaId], references: [id], onDelete: Restrict)

  @@index([userId, status, createdAt])
  @@index([usageQuotaId, status])
  @@index([status, expiresAt])
}

model WebhookEvent {
  id              String    @id @default(cuid())
  provider        String
  providerEventId String
  eventType       String
  status          String    @default("RECEIVED")
  attemptCount    Int       @default(0)
  claimedBy       String?
  claimToken      String?   @unique
  leaseUntil      DateTime?
  resourceKey     String?
  providerEventAt DateTime?
  resourceVersion String?
  payloadCiphertext String?
  receivedAt      DateTime  @default(now())
  processedAt     DateTime?
  errorCode       String?

  @@unique([provider, providerEventId])
  @@index([status, leaseUntil])
  @@index([status, receivedAt])
}

model WebhookResourceState {
  id                    String    @id @default(cuid())
  provider              String
  resourceKey           String
  lastProviderEventAt   DateTime?
  lastResourceVersion   String?
  lastEventId           String?
  updatedAt             DateTime  @updatedAt

  @@unique([provider, resourceKey])
  @@index([provider, updatedAt])
}
```

`NotificationJob.lastError` ต้องเป็น sanitized error code ไม่ใช่ provider response ที่อาจมี email/token; `storageKey` ของ export ต้องเป็น private object-storage key ไม่ใช่ public URL

`NotificationJob` ใช้ state machine `PENDING -> PROCESSING -> SENT` หรือ `PENDING -> PROCESSING -> RETRY -> FAILED/CANCELED`; worker claim ด้วย transaction/conditional update สร้าง `claimToken` ใหม่ทุกครั้งและตั้ง `leaseUntil` เพื่อกู้ job ที่ worker ตายกลางทาง `payloadCiphertext`/`recipientCiphertext` ต้องเข้ารหัสด้วย application key หรือ KMS และไม่ใช้เก็บ plaintext token/email ใน log

`DataRequest.requestKey` ใช้ dedupe request ต่อ user/type/time window, `downloadTokenHash` ใช้ตรวจ one-time download และต้อง mark `downloadedAt` แบบ atomic; token raw ไม่เก็บใน database `ExportDeletion` เป็น durable manifest แยกจาก DataRequest เพื่อให้ลบ object ได้แม้ request row ถูก purge

`AudioArtifact` เป็น canonical owner/lifecycle record ของ output file; `ProcessingJob` และ `ProjectRecord` ต้องผูกกับ artifact เดียวกันใน transaction เดียว ไม่ให้แต่ละ model ถือ owner/file ID คนละชุด `ownerType` ต้องเป็น `USER`, `GUEST` หรือ `PURGED` แบบ mutually exclusive: `USER` ต้องมี `userId` และไม่มี `ownerKeyHash`, `GUEST` ต้องมี `ownerKeyHash` และไม่มี `userId`, ส่วน `PURGED` ต้องไม่มี owner ทั้งคู่และใช้เฉพาะ cleanup tombstone ที่ยังรอ manifest; migration ต้องเพิ่ม SQL check constraint เพราะ Prisma schema อย่างเดียวบังคับ XOR นี้ไม่ได้

เพิ่ม `artifactId String? @unique` + `artifact AudioArtifact?` ใน `ProjectRecord` ที่มี output file และเพิ่ม `reservations QuotaReservation[]` ใน `UsageQuota`; `ProjectRecord.fileId` เดิมเป็น legacy field ที่ต้อง backfill แล้วค่อยเลิกใช้หลัง artifact ownership ผ่าน verification

download/playback ต้องตรวจ `AudioArtifact.status`, owner, action และ expiry ก่อนส่งไฟล์; cleanup ใช้ `ArtifactCleanup` manifest/lease เป็น control plane ไม่ลบตาม mtime อย่างเดียว

`QuotaReservation` ใช้ state machine `RESERVED -> CONSUMED` หรือ `RESERVED -> REFUNDED/EXPIRED` และมี `UNKNOWN -> RECONCILING -> CONSUMED/REFUNDED` สำหรับ timeout ที่ backend outcome ไม่ชัดเจน; ผูก `usageQuotaId` กับ period ที่ถูกหักโดยตรง, `operationId` เป็น unique idempotency key, ใช้ `reconciliationDeadline`/attempt metric และ expiry reconciler ห้ามเปลี่ยน `UNKNOWN/RECONCILING` เป็น `EXPIRED` หรือ refund จนกว่าจะมีหลักฐานจาก job/artifact; deadline ที่หมดต้องเข้า manual review ไม่ใช่ silent expiry

`WebhookEvent` ใช้ state machine `RECEIVED -> PROCESSING -> PROCESSED` หรือ `RECEIVED/PROCESSING -> RETRY -> FAILED`; duplicate ที่ `PROCESSED` แล้วตอบ 2xx โดยไม่ทำ mutation ซ้ำ, ส่วน row ที่ยังไม่ processed และ lease หมดอายุต้องถูก reclaim ไม่ใช่ตอบสำเร็จทิ้ง event `claimToken` เป็น fencing token และ `providerEventAt/resourceVersion/resourceKey` ใช้ตรวจ ordering กับ last-applied marker ของ subscription/payment resource

ใช้ `WebhookResourceState` keyed by `(provider, resourceKey)` แทน marker เดียวบน Subscription เพื่อรองรับหลาย charge/schedule/resource; payment webhook service ต้อง update resource marker กับ business state ใน transaction เดียว และ reject/ignore event ที่เก่ากว่าโดยใช้ provider-specific version comparison แล้วจึงไม่ให้ out-of-order delivery downgrade subscription

เพิ่ม marker ระดับ subscription เช่น `Subscription.lastWebhookSequence/lastWebhookEventAt` หรือ `BillingSubscriptionState` แบบ one-to-one; ทุก event ที่ map ได้กับ subscription ต้อง lock subscription row และ compare global provider sequence/resource version ภายใน transaction เดียวกับ resource marker เพื่อกัน charge/schedule คนละ resource ที่มาถึงสลับกันแล้ว mutate subscription ผิดลำดับ

การ update ต้องใช้ `SELECT ... FOR UPDATE`/serializable transaction หรือ atomic compare-and-set ที่ตรวจ `provider`, `resourceKey`, version และ event timestamp ในเงื่อนไขเดียวกัน; ถ้า `resourceKey`, provider version, provider sequence หรือ provider event time ที่จำเป็นหาย/parse ไม่ได้ ให้ fail closed/retry โดยไม่ทำ billing mutation ไม่ใช้ `receivedAt` แทน provider ordering และไม่ใช้ metadata จาก webhook เป็น owner authority

`ProcessingJob.phase` เป็น vocabulary เดียวของ operation และ `AudioArtifact.status` เป็น output lifecycle ที่ map กันดังนี้: `RESERVED/UPLOAD_INTENT_ISSUED/UPLOADING/PROCESSING -> RUNNING`, `SUCCEEDED -> COMPLETED`, `FAILED -> FAILED`, `CANCELED -> CANCELED`, `UNKNOWN -> UNKNOWN`, owner purge -> `PURGED`; ห้ามมี business logic ที่อ่าน `ProcessingJob.status` แยกจาก phase

`ProcessingJob` เป็น durable operation state ด้วย `phase`, `bodyDigest`, `bytesAcceptedAt`, `claimToken`, `leaseUntil` และ `updatedAt`; ทุก transition ใช้ conditional update จาก phase เดิม + claim token, ทำให้ upload retry, backend timeout และ account purge ตรวจ/ตัดงานเดียวกันได้โดยไม่พึ่ง in-memory `job_manager` อย่างเดียว

`bytesAcceptedAt` ถูกเขียน atomically ตอน backend claim chunk แรกก่อนเริ่ม stream; `bodyDigest` จะเขียนเมื่อ stream จบและ transition `UPLOADING -> PROCESSING` สำเร็จ ไม่อ้างว่า digest กับ first-byte อยู่ transaction เดียวกัน หาก client disconnect หลังรับ bytes แต่ก่อน digest เสร็จ ให้ phase เป็น `UNKNOWN`, fence token เดิม, block retry ด้วย operation เดิมจน reconciler ตรวจ/ยกเลิก backend job; reissue token ได้เฉพาะ operation ที่ยังไม่มี `bytesAcceptedAt`

`UNKNOWN` ต้องตั้ง `reconciliationDeadline` และ `reconcileAttemptCount`; worker retry status/cancel ด้วย backoff จนได้ terminal evidence. เมื่อ deadline หมดให้ย้ายงานเข้า manual-review/dead-letter พร้อม alert, invalidate token/grant และห้ามคืน quota อัตโนมัติจนตรวจ outcome ได้ ไม่ปล่อย partial upload หรือ reservation ค้างแบบไม่มี owner/metric

### 4.6 Consent model

`emailNotifications` เป็น preference ไม่ใช่หลักฐาน consent สำหรับ marketing:

```prisma
model UserConsent {
  id          String    @id @default(cuid())
  userId      String?
  consentType String
  policyVersion String
  grantedAt   DateTime
  revokedAt   DateTime?
  ipHash      String?
  user        User?     @relation(fields: [userId], references: [id], onDelete: SetNull)

  @@index([userId, consentType, grantedAt])
}

model AdminFactor {
  id                    String    @id @default(cuid())
  userId                String
  type                  String
  secretCiphertext      String?
  credentialId          String?
  credentialPublicKey   String?
  enabledAt             DateTime?
  disabledAt            DateTime?
  createdAt             DateTime  @default(now())
  user                  User      @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, type, disabledAt])
}
```

`consentType` อย่างน้อยต้องมี `TERMS`, `PRIVACY`, `ESSENTIAL_EMAIL`, `MARKETING_EMAIL`; essential security/billing email ยกเลิกไม่ได้ด้วย preference เดียว

### 4.7 OAuth token storage

ปัจจุบัน `Account.access_token`, `refresh_token` และ `id_token` เป็น string fields ตาม PrismaAdapter และอาจถูกเก็บ plaintext จึงต้องมี policy แยก:

- ถ้า provider/flow ไม่ต้องใช้ refresh หรือ reconnect ให้ไม่ persist token และตั้ง field เป็น `null` หลัง login
- ถ้าต้อง persist ให้ใช้ encrypted adapter wrapper เช่น `lib/encrypted-oauth-adapter.ts` เข้ารหัส AES-256-GCM/envelope encryption ก่อนเรียก Prisma และ decrypt เฉพาะ server-side provider refresh; ciphertext format ต้องมี `keyVersion:nonce:ciphertext:tag`
- `APP_ENCRYPTION_KEY`/KMS key ต้องแยกจาก `NEXTAUTH_SECRET` และ `PII_HASH_SECRET`; key rotation ใช้ current+previous decrypt window แล้ว re-encrypt ทุก Account record ก่อนปิด key เก่า
- migration ต้องอ่าน token เดิมใน maintenance/worker ที่จำกัดสิทธิ์, encrypt แล้ว verify decrypt/length/count, จากนั้นห้ามมี plaintext token เหลือใน database หรือ backup ที่ใช้งานต่อ; failure ต้องหยุด migrationและไม่ลบต้นฉบับจนกว่าจะ verify ครบ
- ก่อน cutover ให้สร้าง encrypted snapshot ที่ใช้ KMS/key version เดียวกันเป็น rollback artifact; ห้ามใช้ plaintext database snapshot เป็น rollback path และห้ามลบ old key จน encrypted snapshot/restore drill ผ่าน
- access/refresh/id token ไม่ถูก export, audit, operational log หรือส่งไป client; revoke provider token แบบ best effort ตอน unlink/delete

Files: `lib/encrypted-oauth-adapter.ts`, `lib/token-encryption.ts`, `prisma/migrations/20260802000200_encrypt_oauth_tokens/`, `tests/oauth-token-encryption.test.ts`

---

## 5. System Plan 1: Structured Operational Logging & Request Tracing

### เป้าหมาย

ทำให้ทุก service มี log ที่ค้นหาและ alert ได้ โดยไม่ใช้ database เป็นที่เก็บทุก request และไม่รั่วข้อมูลส่วนบุคคลหรือ secret

### Files ที่เกี่ยวข้อง

- Create: `lib/logger.ts`
- Create: `lib/request-context.ts`
- Create: `lib/log-redaction.ts`
- Modify: `middleware.ts`
- Modify: `lib/auth.ts`
- Modify: `backend/utils/auth_guard.py`
- Modify: `backend/cleanup_task.py`
- Modify: `backend/main.py`
- Create: `backend/utils/logging_config.py`
- Create: `backend/middleware/request_context.py`
- Test: `tests/logger.test.ts`
- Test: `backend/tests/test_logging.py`

### Design

Operational log หนึ่งรายการมี shape มาตรฐาน:

```json
{
  "timestamp": "2026-08-02T12:00:00.000Z",
  "level": "info",
  "service": "web",
  "event": "api.request.completed",
  "request_id": "req_01...",
  "route": "/api/account/profile",
  "method": "PUT",
  "status": 200,
  "duration_ms": 148,
  "user_id_hash": "hmac...",
  "ip_hash": "hmac..."
}
```

FastAPI ใช้ middleware สร้าง/propagate request ID, วัด duration และ log status; Next.js ใช้ helper เดียวกันกับ route ที่มีความเสี่ยงสูงก่อน แล้วขยายไป API ทั้งหมด ห้าม log request/response body แบบ generic

### Event names ขั้นต่ำ

- `api.request.started`, `api.request.completed`, `api.request.failed`
- `auth.login.success`, `auth.login.failed`, `auth.rate_limited`
- `security.csrf_rejected`, `security.backend_token_rejected`
- `audio.job.started`, `audio.job.completed`, `audio.job.failed`
- `billing.webhook.accepted`, `billing.webhook.rejected`
- `system.audit_write_failed`, `system.notification_failed`, `system.cleanup_failed`

### Error handling

- Client ได้ generic error เดิม; รายละเอียด exception ไป server log พร้อม `request_id`
- Redactor ต้องทำงานก่อน logger ทุกครั้ง และมี unit test สำหรับ key แบบ case-insensitive เช่น `password`, `authorization`, `cookie`, `token`, `secret`
- ถ้า logger/collector ล่ม ให้ route ยังทำงานต่อ แต่ต้องนับ metric/เขียน fallback stderr และแจ้ง alert
- ถ้า `AuditLog` เขียนไม่ได้ ให้ operational log เป็น `system.audit_write_failed` และห้ามกลืนโดยไม่มีสัญญาณเตือน
- ต้อง inventory และแทนที่ raw log เดิมใน `lib/auth.ts` (NextAuth variadic error), `backend/utils/auth_guard.py` (IP), `backend/cleanup_task.py` (full path) และทุก router exception path ก่อนผ่าน acceptance
- logger/reverse proxy ต้อง log เฉพาะ normalized route template ไม่ใช่ full URL/query; query keys `token`, `code`, `state`, `session`, `email` และ `password` ต้อง redacted ก่อนส่ง collector/access log

### Acceptance criteria

- ทุก request ที่ไป FastAPI มี `request_id` ใน log และ response header
- ทดสอบแล้วว่า password, token, cookie, raw email/IP และ audio content ไม่ปรากฏใน log
- ค้นเหตุการณ์จาก `request_id` ข้าม Next.js และ FastAPI ได้
- 5xx log มี error class/stack ฝั่ง server แต่ไม่มี secret หรือ request body
- log format เป็น JSON parse ได้ทั้ง development และ production

### Tests

- logger redaction snapshot tests
- request ID: client supplied valid/invalid/missing
- middleware duration/status/error path
- FastAPI exception handler ยังคืนข้อความ generic และ log `request_id`

---

## 6. System Plan 2: Audit Log / User Activity Trail

### เป้าหมาย

เก็บหลักฐานการเปลี่ยนแปลงที่เกี่ยวกับ account, security, billing, quota, audio lifecycle และ admin โดย audit log ต้องตอบได้ว่าใครทำอะไรกับ resource ใด เมื่อไร ผลเป็นอย่างไร และ request ใดเป็นต้นเหตุ

### Files ที่เกี่ยวข้อง

- Modify: `prisma/schema.prisma`
- Create: `prisma/migrations/20260802000100_add_audit_log/`
- Create: `lib/audit.ts`
- Create: `lib/audit-events.ts`
- Create: `app/api/account/activity/route.ts`
- Modify: `lib/auth.ts`
- Modify: `app/api/auth/register/route.ts`
- Modify: `app/api/account/route.ts`
- Modify: `app/api/account/profile/route.ts`
- Modify: `app/api/account/password/route.ts`
- Modify: `app/api/account/providers/route.ts`
- Modify: subscription, quota, history และ webhook routes
- Test: `tests/audit.test.ts`

### Event taxonomy

| Domain | Events |
|---|---|
| Auth | `AUTH_REGISTER`, `AUTH_LOGIN`, `AUTH_LOGIN_FAILED`, `AUTH_LOGOUT`, `AUTH_ACCOUNT_LOCKED`, `AUTH_EMAIL_VERIFIED`, `AUTH_PASSWORD_RESET_REQUESTED`, `AUTH_PASSWORD_RESET_COMPLETED`, `AUTH_PASSWORD_CHANGED`, `AUTH_REAUTH_SUCCESS`, `AUTH_REAUTH_FAILED` |
| Provider/session | `AUTH_PROVIDER_LINKED`, `AUTH_PROVIDER_UNLINKED`, `AUTH_SESSION_REVOKED`, `AUTH_ALL_SESSIONS_REVOKED` |
| Account | `ACCOUNT_PROFILE_UPDATED`, `ACCOUNT_EMAIL_CHANGE_REQUESTED`, `ACCOUNT_EMAIL_CHANGED`, `ACCOUNT_PREFERENCES_UPDATED`, `ACCOUNT_EXPORT_REQUESTED`, `ACCOUNT_EXPORT_COMPLETED`, `ACCOUNT_DELETION_REQUESTED`, `ACCOUNT_ANONYMIZED` |
| Billing | `SUBSCRIPTION_CHECKOUT_STARTED`, `SUBSCRIPTION_ACTIVATED`, `SUBSCRIPTION_CANCELED`, `PAYMENT_SUCCEEDED`, `PAYMENT_FAILED`, `PAYMENT_WEBHOOK_REJECTED` |
| Audio/quota | `AUDIO_JOB_STARTED`, `AUDIO_JOB_COMPLETED`, `AUDIO_JOB_FAILED`, `AUDIO_DOWNLOAD`, `AUDIO_HISTORY_DELETED`, `QUOTA_CONSUMED`, `QUOTA_REFUNDED` |
| Security/admin | `SECURITY_RATE_LIMITED`, `SECURITY_CSRF_REJECTED`, `SECURITY_BACKEND_TOKEN_REJECTED`, `ADMIN_USER_VIEWED`, `ADMIN_AUDIT_VIEWED`, `ADMIN_USER_SUSPENDED`, `ADMIN_USER_UNSUSPENDED`, `ADMIN_SESSION_REVOKED`, `ADMIN_TIER_CHANGED` |

### `lib/audit.ts` interface

กำหนด interface กลางเพื่อไม่ให้ route สร้าง metadata เองแบบไม่มีมาตรฐาน:

```typescript
type AuditOutcome = "SUCCESS" | "FAILURE" | "DENIED";
type ActorType = "USER" | "SUPPORT" | "ADMIN" | "SYSTEM" | "ANONYMOUS";

type AuditInput = {
  event: string;
  outcome: AuditOutcome;
  actorType: ActorType;
  actorId?: string;
  userId?: string;
  targetType?: string;
  targetId?: string;
  requestId?: string;
  ip?: string;
  userAgent?: string;
  metadata?: Record<string, string | number | boolean | null>;
};

export async function recordAuditEvent(input: AuditInput): Promise<void>;
```

`recordAuditEvent` ต้อง normalize/hash PII, filter metadata ด้วย allowlist และกำหนด `retentionUntil` ตาม event ก่อนเรียก Prisma

### Logging rules

- Login สำเร็จ/ล้มเหลวต้อง log แม้ user lookup ไม่พบ; ถ้าไม่ทราบ `userId` ให้ใช้ `ANONYMOUS` + email HMAC ใน metadata
- Log password change/reset สำเร็จ แต่ไม่เก็บ password หรือ token
- Log profile update เฉพาะ field names ที่เปลี่ยน เช่น `changed_fields: "name,email"`
- Log tier/payment status แต่ไม่ log payment token/card data
- Audio event เก็บ `action`, `file_id`, `project_record_id`; ไม่เก็บ WAV/name เต็ม
- Account deletion ต้องสร้าง audit ก่อน anonymize และสร้าง completion event แบบ `SYSTEM`
- Activity API แสดงเฉพาะ event ที่เป็น user-visible; ไม่เปิด raw IP/hash, internal metadata หรือ admin/security investigation fields ให้ user

### Retention defaults

- Security/auth events: 365 วัน
- Billing/admin events: 7 ปีหรือตามข้อกำหนดทางบัญชี/กฎหมายที่องค์กรกำหนด
- Audio activity: 90 วัน หรือเท่ากับ project history policy
- Rate-limit/failed-attempt detail: 30 วัน
- `retentionUntil` เป็น required field; migration ต้อง backfill เป็น `createdAt + 365 วัน` สำหรับ legacy rows และไม่มีค่า `NULL` ที่ cleanup job ต้องตีความเอง
- หลัง account anonymize ให้ตัด `userId`, `actorId` ที่ชี้ user, `targetId` ที่ชี้ user, `userAgent`, `ipHash` และ metadata ที่ระบุตัวบุคคลออก แต่คง event/เวลา/outcome/request ID ตาม retention
- `piiHashKeyVersion` ต้องเก็บคู่กับ hash; ตอน rotate ให้รองรับ key เก่าเพื่อค้นหาในช่วง migration แล้วหยุดเขียนด้วย key เก่าเมื่อ dual-read window จบ

### Acceptance criteria

- Login, register, logout, password/provider/profile/subscription/quota/delete ทุกเส้นทางมี audit event ที่ตรงกัน
- Audit record ของ user หนึ่งไม่สามารถถูกอ่านหรือแก้โดย user อื่น
- การเขียน audit ล้มเหลวมี operational alert และไม่ทำให้ response ส่งข้อมูลลับ
- `GET /api/account/activity` paginate ได้, จำกัด page size, sort ล่าสุดก่อน และไม่คืน secret
- Admin สามารถค้น audit ตาม event/time/user แต่ทุก admin read มี event `ADMIN_USER_VIEWED` หรือ `ADMIN_AUDIT_VIEWED`

### Tests

- event mapping ของทุก route สำคัญ
- metadata redaction และ retention assignment
- ownership/authorization ของ activity endpoint
- account deletion ไม่ cascade ลบ audit ที่ต้อง retain
- duplicate webhook/idempotent action ไม่สร้าง audit success ซ้ำโดยไม่มี dedupe key
- rate-limit/CSRF/backend-token rejection map ไป `SECURITY_*` event เดียวกันทั้ง Next.js และ FastAPI

---

## 7. System Plan 3: Persistent Rate Limiting & Brute-force Protection

### เป้าหมาย

เปลี่ยน rate limit จาก in-memory เป็น distributed/persistent state และป้องกัน login abuse โดยไม่เปิดเผยว่า email ใดมี account และไม่ทำให้ attacker lock account ของผู้อื่นได้ง่ายเกินไป

### Files ที่เกี่ยวข้อง

- Modify: `middleware.ts`
- Modify: `lib/rate-limit.ts`
- Create: `lib/redis.ts`
- Create: `lib/auth-abuse.ts`
- Modify: `lib/auth.ts`
- Modify: `app/api/auth/register/route.ts`
- Modify: `app/api/account/password/route.ts`
- Create/modify: forgot/reset verification routes
- Modify: `prisma/schema.prisma` (`failedLoginCount`, `lockedUntil`)
- Test: `tests/rate-limit.test.ts`

### Recommended implementation

เนื่องจาก `middleware.ts` อาจทำงานใน Edge runtime ให้เลือก Redis client ที่เข้ากับ runtime เดียวกันทั้งระบบ โดย default ใช้ `@upstash/redis`; ถ้า deploy เป็น Node server ที่เข้าถึง Redis โดยตรง ให้ใช้ `ioredis` และย้าย logic ที่ต้องใช้ TCP ไป route runtime ที่เหมาะสม ไม่ติดตั้งสอง client โดยไม่จำเป็น

### Limit matrix

| Key | Limit | Window | Action เมื่อเกิน |
|---|---:|---:|---|
| `login:ip` | 10 | 1 นาที | 429 + audit `AUTH_LOGIN_FAILED`/`RATE_LIMITED` |
| `login:identifier` | 5 | 15 นาที | generic auth failure; ไม่บอกว่า locked เพราะอะไร |
| `register:ip` | 3 | 1 ชั่วโมง | 429 |
| `password-change:user` | 5 | 1 ชั่วโมง | 429 + re-auth required |
| `password-reset:ip` | 5 | 1 ชั่วโมง | generic 202 |
| `password-reset:identifier` | 3 | 24 ชั่วโมง | generic 202 |
| `resend-verification:user` | 3 | 1 ชั่วโมง | generic 202 |
| `account-mutation:user` | 60 | 1 นาที | 429 |
| `backend:sub` | ตาม tier/concurrency | 1 นาที | 429/403 ตาม policy |

Keys ที่มี email ต้องใช้ HMAC email ไม่ใช้ email plain ใน Redis key เพื่อไม่ให้ข้อมูลส่วนบุคคลค้างใน infrastructure log

### Lockout policy

- นับ failed credential login แบบ atomic ต่อ user เมื่อพบ user แต่ password ไม่ถูกต้อง และเก็บ `loginFailureWindowStartedAt` เพื่อคำนวณ rolling window ได้จริง
- อย่าใช้ hard account lock จาก password failure อย่างเดียวใน P0 เพราะ attacker สามารถ lock victim ได้; ใช้ per-IP + per-identifier progressive delay และ soft lock ที่หมดอายุ/กู้ผ่าน verified password-reset ได้
- `lockedUntil` ใช้กับ risk engine ที่มีสัญญาณหลายด้านหรือ admin/system suspension เท่านั้น; หากเปิด account-level soft lock ใน phase ถัดไปต้องมี reset path และ notification เสมอ
- ห้ามบอก client ว่า email ถูกล็อก; response ยังคง generic `Invalid email or password`
- reset counter เมื่อ login สำเร็จ, password reset สำเร็จ หรือ admin unlock
- IP limiter และ identifier limiter ยังทำงานแม้ user ถูกล็อก เพื่อป้องกัน brute-force distribution
- ไม่ lock user จาก OAuth failure หรือ failed request ที่ไม่ผ่าน schema validation
- ทุก lock/unlock มี audit event และ security notification ตาม policy

### Guest quota migration

- ย้าย guest quota จากไฟล์ JSON/raw IP ใน `backend/utils/auth_guard.py` ไป Redis atomic counter ที่ key ด้วย HMAC IP และ daily TTL เมื่อ deploy หลาย process
- ถ้ายัง single-process ให้ไฟล์เป็น compatibility fallback ชั่วคราวเท่านั้น พร้อม bounded size, hashed IP และ metric `guest_quota_degraded`
- guest quota ต้องใช้ `INCR`/expiry แบบ atomic และมี cleanup/retention ไม่สะสม IP เก่าไม่จำกัด

### Redis failure policy

- ถ้า Redis unavailable ให้ใช้ bounded in-process fallback สำหรับ auth endpoints และ log `system.rate_limiter_degraded`
- จำกัด fallback memory และ cleanup expired entries เพื่อไม่ให้ process โตไม่จำกัด
- ถ้า fallback เกิน safe threshold ให้ reject เฉพาะ sensitive auth mutation ด้วย 503 แทนการปล่อยผ่านโดยไม่มี protection
- มี alert เมื่อ Redis unavailable ต่อเนื่องเกิน 60 วินาที

### Acceptance criteria

- Rate limit ใช้ร่วมกันได้เมื่อรัน Next.js อย่างน้อยสอง process
- Counter เพิ่มแบบ atomic และไม่ reset จาก process restart
- Concurrent login failures ไม่ทำให้ counter สูญหาย
- Error response ไม่แยก `unknown email`, `wrong password`, `locked account`
- Test ยืนยันว่า attacker ไม่สามารถส่ง `X-Forwarded-For` เพื่อ bypass ถ้า reverse proxy ไม่ได้ trust header นั้น

### Tests

- Redis mock: increment/expiry/atomic behavior
- middleware IP limiter และ route-level identifier limiter
- lockout threshold, reset on success, expiry, admin unlock
- Redis outage fallback
- race test สำหรับ failed login 10 requests พร้อมกัน
- account DoS test: repeated failures ไม่สร้าง permanent lock และ verified reset กู้ access ได้
- guest quota concurrent increment, daily expiry, hashed key และ multi-process behavior

---

## 8. System Plan 4: Backend JWT/Service-token Validation

### เป้าหมาย

ปิดช่องโหว่ที่ client เรียก FastAPI โดยส่ง `X-User-ID`/`X-User-Tier` ปลอม แล้วเปลี่ยนเป็น identity ที่ backend ตรวจสอบ cryptographically ได้

### ทางเลือกและข้อสรุป

| ทางเลือก | ข้อดี | ข้อเสีย | ข้อสรุป |
|---|---|---|---|
| Trust custom headers | แก้เร็ว | ปลอมได้เต็มที่ | ห้ามใช้ใน production |
| Browser ขอ signed token แล้วเรียก FastAPI ตรง | ไม่ต้อง proxy file 100MB ผ่าน Next.js | backend ต้อง public และ token อยู่ใน browser ระยะสั้น | ใช้เป็น fallback |
| Next.js BFF/private network | secret ไม่อยู่ browser, policy อยู่จุดเดียว | proxy ต้อง stream upload และเพิ่ม bandwidth ฝั่ง web | **แนะนำเป็นค่าเริ่มต้น** |

### Files ที่เกี่ยวข้อง

- Create: `lib/backend-token.ts`
- Create: `app/api/backend/[...path]/route.ts` หรือ route proxy แยกตาม endpoint
- Modify: `lib/hooks/useAudioProcessor.ts`
- Modify: `lib/config.ts`
- Modify: `backend/utils/auth_guard.py`
- Create: `backend/utils/service_auth.py`
- Modify: `backend/routers/stems.py`
- Modify: `backend/routers/audio_ops.py`
- Modify: `backend/main.py`
- Modify: `backend/requirements.txt`
- Test: `tests/backend-token.test.ts`
- Test: `backend/tests/test_service_auth.py`

### Token claims

```json
{
  "iss": "harmoniq-web",
  "aud": "harmoniq-audio-backend",
  "sub": "user-id",
  "tier": "BASIC",
  "scope": "audio:upload",
  "phase": "UPLOAD_INTENT_ISSUED",
  "operation_id": "op_01...",
  "reservation_id": "quota_01...",
  "job_id": "job_01...",
  "job_claim_token": "fence_01...",
  "jti": "one-time-id",
  "iat": 1722600000,
  "exp": 1722600300,
  "kid": "backend-key-2026-08"
}
```

ข้อกำหนด:

- อายุ token ไม่เกิน 5 นาที และ clock skew ไม่เกิน 30 วินาที
- ตรวจ algorithm allowlist, `iss`, `aud`, `exp`, `iat`, `scope`, `sub` และ tier allowlist
- token ของ BFF ต้องผูกกับ `operation_id`, `reservation_id`, `job_id`, `job_claim_token`, `phase` และ owner; FastAPI ต้อง reject token ที่ไม่ตรงกับ job/reservation/claim fence หรือใช้กับ operation อื่น
- BFF ไม่สามารถใส่ final body digest ใน token ก่อน stream ไฟล์ 100MB: backend ต้อง claim upload intent แบบ one-time, stream แล้วคำนวณ digest, บันทึก digest ใน operation state และเปลี่ยน `UPLOADING -> PROCESSING`; retry ใช้ status/idempotency ของ operation เดิม ไม่ replay token ที่ถูก consume
- BFF default อยู่บน private network และ browser ไม่เห็น service token; ทุก FastAPI heartbeat/status/cancel callback ต้องส่ง service signature + `operation_id` + `job_claim_token` และ Next.js ต้องทำ conditional update ตรง fence เดิม ถ้าเปิด direct fallback ต้องส่ง `Content-Digest` ผ่าน preflight ก่อนออก token หรือใช้ upload-intent state เดียวกัน, claim `jti` ผ่าน Redis atomic replay record และยอมรับ retry เฉพาะ operation ที่ยังไม่รับ bytes/owner ตรงกัน
- key rotation รองรับ current + previous key ด้วย `kid`; ลบ previous หลัง token เก่าหมดอายุ
- ใช้ `BACKEND_AUTH_SECRET` แยกจาก `NEXTAUTH_SECRET`; production ที่มีหลาย service ให้ย้ายเป็น RS256/JWKS
- ถ้ามี `Authorization` แต่ token invalid/expired ให้ 401; ห้าม fallback เป็น guest
- guest ที่ไม่มี `Authorization` ใช้ guest quota เดิมได้

### BFF flow

```text
Browser -> POST /api/backend/separate (cookie)
Next.js -> requireSession + verify email/status + reserve quota + create AudioArtifact/ProcessingJob
Next.js -> sign short-lived UPLOAD_INTENT_ISSUED token from server secret, bound to operation/job/reservation/claim fence
Next.js -> stream multipart request to private FastAPI
FastAPI -> verify claim fence, atomically claim jti, mark first-byte acceptance, stream + compute digest, persist phase, then process audio
FastAPI -> callback status/heartbeat/digest/cancel result with the same claim fence + return file id for the same artifact
Next.js -> consume reservation on terminal success or reconcile before refund
Browser <- sanitized response
```

`phase = UPLOAD_INTENT_ISSUED` ใน token หมายถึง transition ที่ token ได้รับอนุญาตให้ทำเท่านั้น: FastAPI ต้องตรวจว่า persisted `ProcessingJob.phase` ยังเป็น `UPLOAD_INTENT_ISSUED` แล้ว atomically เปลี่ยนเป็น `UPLOADING` ตอน claim แรก; callback หลังจากนั้นไม่ใช้ token phase เพื่อเปิด transition ใหม่ แต่ต้อง match `job_claim_token`/operation fence เดิมทุกครั้ง

ห้ามให้ Next.js buffer ไฟล์ 100MB ทั้งก้อนใน memory; ใช้ streaming และกำหนด body size/timeout ที่ reverse proxy และ FastAPI ให้ตรงกับ limit 100MB

### Artifact authorization / IDOR prevention

ทุก endpoint ที่คืนไฟล์หรือผลลัพธ์ต้องอยู่ใน authorization matrix เดียวกัน:

| Resource | Browser entry point | Server-side check |
|---|---|---|
| User project download/playback | Next.js BFF `/api/audio/files/...` | `AudioArtifact.userId` ตรงกับ session, `ProjectRecord.artifactId`/`ProcessingJob.artifactId` ตรงกับ artifact, action และ expiry ถูกต้อง |
| Guest download/playback | BFF พร้อม signed guest-session cookie | `AudioArtifact.ownerType = GUEST` และ `ownerKeyHash` ตรงกับ guest session, status เป็น terminal และยังไม่หมดอายุ |
| Backend internal file route | private network + bearer token `audio:read` | token subject/job owner/file ID/action/expiry ตรงกัน |
| Unknown/expired/deleted artifact | ทุกช่องทาง | 404 generic; ไม่บอกว่า resource มีอยู่หรือเป็นของใคร |

ห้ามเปิด public `/download/{file_id}`, `/separated/...`, `/karaoke/...` ให้ browser bypass BFF โดยตรง ถ้าต้องรองรับ direct fallback ให้ใช้ short-lived file grant ที่มี `scope=audio:read`, `file_id`, `action`, owner claim, `exp` และ `jti`; grant ของ browser fallback ต้องใช้ Redis `SETNX`/idempotency binding หรือปิด fallback แล้วบังคับ BFF-only

ต้อง inventory และครอบคลุม route ปัจจุบันทั้งหมด ได้แก่ `GET /download/{file_id}`, `GET /separated/{file_id}/{filename}`, `GET /karaoke/{file_id}`, export/process download routes และทุก route ที่รับ `file_id` จาก `lib/hooks/useAudioProcessor.ts`; ห้ามแก้เฉพาะ `/download` แล้วปล่อย derived endpoint เป็น public

`GET /karaoke/{file_id}` ปัจจุบันมี side effect สร้าง/ประมวลผลไฟล์ จึงต้องย้าย generation ไป `POST /api/audio/karaoke` ที่ผ่าน quota/owner/CSRF-or-bearer checks หรือ precompute karaoke artifact ตอน separation แล้วให้ GET เหลือ read-only playback/download เท่านั้น ทุก derived endpoint ต้องอยู่ใน artifact matrix เดียวกัน

### Quota reservation and replay protection

- ก่อนสร้าง backend token ให้ `POST /api/quota/reserve` สร้าง `QuotaReservation` ด้วย `operationId` ที่ unique, `usageQuotaId` ของ period ปัจจุบัน และหัก quota แบบ conditional atomic ใน transaction เดียวกับสร้าง `AudioArtifact`/`ProcessingJob`
- operation state ใช้ `RESERVED -> UPLOAD_INTENT_ISSUED -> UPLOADING -> PROCESSING -> SUCCEEDED/FAILED/UNKNOWN`; `jti` ของ upload intent ใช้ครั้งเดียว, retry หลังรับ bytes ใช้ `GET status/operationId` หรือ server-issued intent ใหม่เฉพาะเมื่อ state ยังไม่รับ bytes และไม่สร้าง reservation/artifact ใหม่
- `bytesAcceptedAt` ถูกเขียนใน conditional claim ตอนรับ chunk แรก และ `bodyDigest` ถูกเขียนใน conditional transition หลัง stream จบ; duplicate request ที่เห็น phase `UPLOADING/PROCESSING/SUCCEEDED` ต้องคืน operation status หรือ 409 ไม่เริ่มอ่าน bytes ซ้ำ
- `POST /api/quota/consume` เปลี่ยน `RESERVED -> CONSUMED` ได้ครั้งเดียว; `POST /api/quota/refund` เปลี่ยนเป็น `REFUNDED` ได้เฉพาะ reservation เดิมและไม่ decrement ซ้ำ
- ถ้า quota service/Redis/DB ตรวจ reservation ไม่ได้ ให้หยุดก่อน upload/process และคืน generic 503; ห้ามดำเนินงานต่อแบบ fail-open เหมือน path เดิมใน `useAudioProcessor.ts`
- ถ้า backend outcome เป็น `UNKNOWN` ห้าม refund ทันที; reconciler ตรวจ callback/artifact/job state ก่อน แล้วค่อยเลือก consume/refund ตามหลักฐาน
- signed-token direct fallback ใช้ Redis replay record `backend:jti:{jti}` แบบ atomic; BFF-only ไม่ต้องเปิด token ให้ browser และเป็น default production mode

### Migration rules

- ช่วง migration รองรับ signed token ก่อน แต่ `X-User-*` ใช้เพื่อ telemetry ได้เท่านั้น ไม่ใช้ตัดสินสิทธิ์; compatibility flag ต้อง default `false` และ rollback ห้ามเปิด legacy header authority กลับมา
- เพิ่ม feature flag `REQUIRE_BACKEND_TOKEN=true` ใน staging แล้วเปิด production หลังตรวจทุก client path
- หลังเปิดจริงให้ลบ code path ที่อ่าน tier/user ID จาก header และลด CORS ให้เหลือ origin/method/header ที่ใช้จริง
- `validate_request_quota` ต้องรับ principal ที่ verified แล้ว ไม่รับ `user_tier` จาก route caller

### Acceptance criteria

- เปลี่ยน `X-User-Tier: PRO` จาก client แล้วไม่ได้สิทธิ์ PRO
- token ที่แก้ `sub`, `tier`, `exp`, `aud` หรือใช้ `alg=none` ถูก reject
- token หมดอายุถูก reject และไม่กลายเป็น guest
- token scope อื่นใช้ audio process ไม่ได้
- authenticated processing ทุกครั้งมี quota ownership จาก Next.js และ backend log มี verified subject hash
- guest flow และ file validation/cleanup เดิมยังทำงาน
- user A ไม่สามารถใช้ `fileId`, project ID หรือ playback URL ของ user B ได้ แม้รู้ identifier
- quota consume/refund ทำซ้ำหรือ concurrent แล้วไม่เปลี่ยนยอดผิด และ operation เดิม replay ได้เฉพาะตาม idempotency policy

### Tests

- valid/expired/wrong issuer/wrong audience/wrong signature/wrong scope/unknown tier
- key rotation current/previous/unknown `kid`
- missing vs malformed Authorization
- direct backend request spoof header
- file download/playback/export IDOR matrix สำหรับ user, guest, expired และ deleted artifact
- GET derived endpoints ไม่มี side effect; karaoke generation ใช้ POST/precomputed artifact และผ่าน owner/quota checks
- BFF streaming, 100MB boundary, timeout และ quota refund
- quota reservation race, duplicate refund และ service failure ต้อง fail-closed
- upload operation phase/bytesAccepted/bodyDigest transition and fenced retry
- stale FastAPI heartbeat/status/cancel callback with an old `job_claim_token` cannot renew or mutate a reclaimed job

---

## 9. System Plan 5: Session & Device Management

### เป้าหมาย

ให้ผู้ใช้เห็น session ที่ active, revoke ราย device, revoke all other sessions และให้ logout/security event มีผลจริงทันที

### Files ที่เกี่ยวข้อง

- Modify: `lib/auth.ts` (`session.strategy = "database"` และ callback/adapter behavior)
- Create: `lib/auth-status.ts` (`canCreateSession` and protected-status policy)
- Modify: `app/api/auth/[...nextauth]/route.ts` (adapter/session creation context)
- Modify: `types/next-auth.d.ts` (role/status/session fields; ห้ามใช้ `any`)
- Modify: `prisma/schema.prisma` (`Session` metadata)
- Create: migration for database sessions
- Modify: `lib/auth.ts`/`lib/session.ts` สำหรับ current-session lookup และ last-seen throttling
- Create: `app/api/account/sessions/route.ts`
- Create: `app/api/account/sessions/[id]/route.ts`
- Create: account UI session section หลังอ่าน `DESIGN.md`
- Test: `tests/sessions.test.ts`

### Design

- เปลี่ยนจาก JWT เป็น database session เพราะ JWT แบบ stateless revoke ราย device ไม่ได้
- session record แต่ละรายการมี `createdAt`, `lastSeenAt`, `lastAuthenticatedAt`, `deviceLabel`, `userAgent` ที่ truncate แล้ว และ `ipHash`
- `GET /api/account/sessions` คืนข้อมูลที่ปลอดภัยเท่านั้น; ห้ามคืน `sessionToken`
- `DELETE /api/account/sessions/:id` ลบ session เฉพาะของ user ปัจจุบัน
- `DELETE /api/account/sessions?scope=other` ลบทุก session อื่นใน transaction เดียว และคง current session
- `DELETE /api/account/sessions?scope=all` ลบทั้งหมด จากนั้น client sign out
- ทุก revoke สร้าง `AUTH_SESSION_REVOKED` หรือ `AUTH_ALL_SESSIONS_REVOKED`
- update `lastSeenAt` อย่างน้อยทุก 5 นาทีต่อ session เพื่อลด DB writes
- อุปกรณ์ใหม่หรือ login จาก fingerprint ที่ต่างกันสร้าง security notification แต่ไม่ block login โดยอัตโนมัติ
- PrismaAdapter จะยังเก็บ session token ตาม contract ของ NextAuth; จำกัด DB/backup access, encrypt backup/at-rest และห้ามนำ token ออก API หรือ log แทนการพยายามเขียน custom hashing โดยไม่มีแผน adapter ครบถ้วน

### Re-auth integration

ใช้ `Session.lastAuthenticatedAt` แทน `token.reauthAt` สำหรับ destructive action:

- credential password ถูกต้อง: update current session `lastAuthenticatedAt`
- OAuth re-auth สำเร็จ: update current session `lastAuthenticatedAt`
- destructive action ยอมรับเฉพาะ current session ที่ re-auth ภายใน 5 นาที
- ห้ามรับ `reauthAt` จาก client

เมื่อใช้ database session callback ต้องอ่าน `user`/database record ตาม contract ของ PrismaAdapter ไม่อ่าน `token.id`, `token.tier` หรือ role จาก JWT เดิม; `session.user.id`, tier, role และ status ต้องมาจาก DB และ `requireSession` ต้องใช้ allowlist: `ACTIVE` ใช้งานปกติ, `PENDING_EMAIL` เข้าได้เฉพาะ account/verification/recovery ที่อนุญาต, ส่วน `SUSPENDED`, `PENDING_DELETION` และ `ANONYMIZED` ใช้ protected API/audio/admin ไม่ได้ (`ANONYMIZED` ห้ามสร้าง session ใหม่) ยกเว้น endpoint recovery/verification ที่ระบุไว้โดยตรง

สร้าง `canCreateSession(user, context)` เป็น centralized auth-creation gate และเรียกจาก `credentialsAuthorize`, NextAuth `signIn` callback, OAuth account linking และ database-session callback ก่อนสร้าง/คง session: `ACTIVE` ผ่าน, `PENDING_EMAIL` ผ่านเป็น limited session, `SUSPENDED/PENDING_DELETION/ANONYMIZED` reject ด้วย generic auth error; verification/password-recovery endpoints ไม่สร้าง session และเป็น explicit exception แทนการ bypass gate

### Cutover

1. deploy schema fields โดยยังไม่เปลี่ยน strategy
2. deploy code ที่รองรับ database session และหน้า sign-in/sign-out
3. เปลี่ยน strategy เป็น database ใน maintenance window สั้น ๆ
4. invalidate JWT cookie เดิมด้วย cookie name/version เพื่อบังคับ sign-in ใหม่ โดยไม่ rotate `NEXTAUTH_SECRET` จนกว่าจะพ้น rollback window
5. monitor session creation, session lookup latency และ database write rate

Rollback ต้องเป็นการ deploy code กลับพร้อม cookie version ที่ชัดเจน และ restore database snapshot เฉพาะก่อนมี writes ที่ย้อนกลับไม่ได้; ห้ามสมมติว่าเปลี่ยน `session.strategy` กลับแล้ว database-session cookie จะกลายเป็น JWT ได้เอง ผู้ใช้ที่มี cookie จาก strategy ใหม่ต้อง sign in ใหม่หลัง rollback

### Acceptance criteria

- revoke session หนึ่งแล้ว request จาก session นั้นได้ 401 ภายใน request ถัดไป
- revoke all other ไม่กระทบ current session
- account delete/password reset revoke ทุก session
- session list ไม่เปิด token/IP raw/password/provider token
- expired sessions cleanup ได้ตาม schedule
- `SUSPENDED`, `PENDING_DELETION`, `ANONYMIZED` ไม่ได้ normal protected session; `PENDING_EMAIL` ได้เฉพาะ allowlisted routes

### Tests

- database session creation/lookup/revoke
- current vs other/all scope
- revoked session cannot access protected API
- recent re-auth window and expiry
- concurrent revoke + request
- session callback ใน database strategy ไม่พึ่ง JWT-only fields และอ่าน role/status ล่าสุดจาก DB
- credentials/OAuth sign-in และ provider link ไม่สร้าง session ให้ `SUSPENDED/PENDING_DELETION/ANONYMIZED`; verification/recovery exception ไม่สามารถขยายเป็น normal session
- first-time OAuth user without a trusted verified claim is persisted as `PENDING_EMAIL` before session creation; OAuth cannot activate restricted/legacy users

---

## 10. System Plan 6: Email Verification & Email Change

### เป้าหมาย

ทำให้ `User.emailVerified` มีความหมายจริง และป้องกันการเปลี่ยนอีเมลโดยผู้โจมตีที่ยึด session ได้บางส่วน

### Files ที่เกี่ยวข้อง

- Modify: `prisma/schema.prisma`
- Create: migration for `EmailVerificationToken`/`EmailChangeToken`
- Create: `lib/email.ts`
- Create: `lib/token.ts`
- Create: `lib/oauth-provisioning.ts`
- Create: `app/api/auth/verify-email/route.ts`
- Create: `app/api/auth/resend-verification/route.ts`
- Modify: `app/api/auth/register/route.ts`
- Modify: `app/api/account/profile/route.ts`
- Modify: `lib/auth.ts`
- Create: verification UI routes/components หลังอ่าน `DESIGN.md`
- Test: `tests/email-verification.test.ts`

### Token policy

- สร้างด้วย cryptographically secure random bytes อย่างน้อย 32 bytes
- ส่ง raw token ใน HTTPS URL/email เท่านั้น; เก็บ HMAC-SHA-256 hash ที่ database โดยมี server-side pepper
- อายุ verification token 24 ชั่วโมง; resend จะ invalidate token เก่าและมี rate limit
- ใช้ได้ครั้งเดียวด้วย transaction ที่ตรวจ `usedAt IS NULL` และ `expiresAt > now`
- ไม่ใส่ email เต็มใน subject/log; template ใช้ชื่อที่ user อนุญาตหรือ generic greeting

### Registration flow

1. normalize email และตรวจ duplicate แบบ case-insensitive
2. สร้าง user เป็น `PENDING_EMAIL`, `emailVerified = null` และ `emailVerificationRequiredAt = createdAt` โดย route ต้อง set ค่า explicit ไม่พึ่ง schema default `ACTIVE`
3. สร้าง subscription/quota ใน transaction เดิม
4. สร้าง token และ enqueue `EMAIL_VERIFICATION`
5. ส่ง response แบบไม่เปิดเผยว่าการส่ง email สำเร็จหรือไม่
6. session ที่สร้างแล้วต้องถูกจำกัดด้วย verification policy

### Verification policy

- ผู้ใช้ sign in ได้เพื่อไปหน้า verification/resend
- ห้าม checkout, เปลี่ยน email, เปลี่ยน password สำหรับ account ที่ยังไม่ verified
- หลัง `emailVerificationRequiredAt <= now` การประมวลผลเสียงของ authenticated userต้อง verified; guest policy เดิมยังใช้ได้
- OAuth provider ที่คืน verified email อย่างเชื่อถือได้สามารถ set `emailVerified`; Facebook/LINE fallback email เช่น `*_local` ต้องไม่ถือว่า verified
- ทุก security email เช่น password reset/change และ billing receipt เป็น essential ไม่ขึ้นกับ `emailNotifications`

OAuth first-login ต้องผ่าน `provisionOAuthUser` ก่อน session issuance: custom adapter wrapper/`createUser` path ตั้ง `status`, `emailVerified` และ `emailVerificationRequiredAt` จาก provider-specific verified claim; ถ้าหา claim ไม่ได้ให้ `PENDING_EMAIL` ทันที ไม่พึ่ง `User.status` database default `ACTIVE`. `signIn` callback ต้อง re-read/reconcile status ก่อน upsert subscription/quota และเรียก `canCreateSession`; existing restricted user ห้ามถูก activate เพราะ OAuth login

### Legacy-user backfill

ก่อนเปิด enforcement ต้องรัน migration/report ตามลำดับนี้:

1. ตรวจ duplicate email หลัง normalize lowercase/trim และหยุด deployment หากมี collision ที่แก้แบบ deterministic ไม่ได้
2. user เดิมที่ `emailVerified IS NOT NULL` ให้ `status = ACTIVE` เฉพาะเมื่อ status เดิมเป็น `ACTIVE/PENDING_EMAIL`; ต้อง preserve `SUSPENDED/PENDING_DELETION/ANONYMIZED`
3. user เดิมที่ `emailVerified IS NULL` และ status เป็น `ACTIVE/PENDING_EMAIL` ให้ `status = PENDING_EMAIL`, enqueue verification campaign; ต้อง preserve `SUSPENDED/PENDING_DELETION/ANONYMIZED` และไม่สร้าง verification token ให้ `ANONYMIZED`
4. OAuth user จะถูก mark verified ได้เฉพาะเมื่อ provider profile มี verified-email claim ที่ระบุใน provider contract; fallback `*_local` ต้องเข้า `PENDING_EMAIL`
5. user ใหม่ตั้ง `emailVerificationRequiredAt = createdAt`; legacy user ตั้ง `emailVerificationRequiredAt = EMAIL_VERIFICATION_ENFORCEMENT_AT` เพื่อมี grace period ที่ตรวจสอบได้จาก DB
6. เปิด `EMAIL_VERIFICATION_ENFORCEMENT_AT` หลัง campaign/monitoring พร้อม; ก่อนเวลานั้น legacy user แสดง banner/resend และยังใช้ processing ได้เฉพาะใน grace policy ส่วน user ใหม่ยังถูก block
7. เมื่อถึง enforcement time user ที่อยู่ `PENDING_EMAIL` sign in/resend ได้ แต่ checkout, password change, authenticated audio processing และ email change ต้องถูก block ด้วยสถานะ/เวลา requirement จาก DB

การเขียน email ทุก path ต้อง normalize ด้วย canonical function เดียวกัน และ production PostgreSQL ต้องมี lower-case unique index หรือ `citext` หลังตรวจ/แก้ duplicate; Prisma `@unique` เพียงอย่างเดียวไม่รับประกัน case-insensitive uniqueness

### API

| Route | Behavior |
|---|---|
| `POST /api/auth/resend-verification` | คืน 202 แบบ generic, สร้าง token/notification เมื่อ user ยังไม่ verified |
| `GET /api/auth/verify-email?token=...` | hash token, เก็บ encrypted verification intent ใน HttpOnly short-lived cookie แล้ว redirect ไป clean no-token URL แบบ `no-store`, `Referrer-Policy: no-referrer`, ปิด analytics/access-log query; ห้าม consume เพราะ mail scanner/prefetch อาจเรียก GET |
| `POST /api/auth/verify-email` | consume intent/token แบบ one-time แล้ว set `emailVerified`; เปลี่ยน `PENDING_EMAIL -> ACTIVE` ได้เท่านั้น และต้อง preserve `SUSPENDED`, `PENDING_DELETION` หรือ `ANONYMIZED`, audit event ทุกกรณี |
| `POST /api/account/profile/email-change` | re-auth, สร้าง pending email token, ยังไม่แก้ primary email |
| `GET /api/auth/confirm-email-change?token=...` | hash token, เก็บ encrypted intent ใน HttpOnly short-lived cookie แล้ว redirect ไป clean no-token URL แบบ `no-store`, `Referrer-Policy: no-referrer`, ปิด analytics/access-log query |
| `POST /api/auth/confirm-email-change` | consume intent/token แบบ one-time, ตรวจ duplicate, transaction ย้าย email + reset verified state ตาม policy |

### Acceptance criteria

- token ซ้ำ/หมดอายุ/ถูกใช้แล้วใช้ไม่ได้และไม่เปลี่ยน user state
- response resend ไม่เปิดเผยว่า email มีอยู่หรือไม่
- email change ไม่ทำให้ account ถูกยึดด้วย session เดิมอย่างเดียว
- GET จาก email scanner ไม่ consume token; token ไม่ค้างใน browser history/referrer/analytics และการ consume ทำผ่าน POST ที่มี CSRF/origin protection
- email change สำเร็จต้องแจ้ง old/new address ตาม policy, invalidate pending/reset tokens และ revoke sessions อื่นเมื่อเป็น security-sensitive change
- verification/email-change GET redirects และ clean confirmation pages ต้องใช้ `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, ปิด analytics และ redacted query logging เหมือน reset flow
- verified state ไม่ถูก set จาก client หรือ query parameter ที่ไม่มี token และ verification ไม่สามารถปลด `SUSPENDED/PENDING_DELETION` ได้
- provider/local fallback email ไม่ถูก mark verified โดยอัตโนมัติ

### Tests

- registration creates token and pending state
- valid/expired/replayed token
- resend invalidates old token and is rate limited
- duplicate email race during confirmation
- OAuth verified/unverified profile behavior
- new user blocked immediately; legacy user allowed only before `emailVerificationRequiredAt`, then blocked from checkout/password change/backend processing
- GET scanner/prefetch cannot consume token and query token is redacted from request logs

---

## 11. System Plan 7: Password Reset & Account Recovery

### เป้าหมาย

ให้ผู้ใช้กู้ account ได้โดยไม่เปิดเผย account existence และไม่ทำให้ reset token/password ถูกขโมยจาก database หรือ log

### Files ที่เกี่ยวข้อง

- Create: `app/api/auth/forgot-password/route.ts`
- Create: `app/api/auth/reset-password/route.ts`
- Create: reset UI routes/components หลังอ่าน `DESIGN.md`
- Create/modify: `lib/token.ts`, `lib/email.ts`, `lib/auth-abuse.ts`
- Modify: `lib/auth.ts`
- Modify: `prisma/schema.prisma`
- Test: `tests/password-reset.test.ts`

### Flow

#### Forgot password

1. รับ email, normalize และ rate limit ด้วย IP + HMAC identifier
2. ตอบ `202 { message: generic }` เสมอ ไม่ว่า user มีหรือไม่
3. ถ้ามี credentials user ให้ revoke reset token เก่า, สร้าง token ใหม่ expiry 1 ชั่วโมง และ enqueue email
4. audit เป็น `AUTH_PASSWORD_RESET_REQUESTED` โดยไม่ log raw email

#### Reset password

1. รับ token + new password; ห้ามรับ email เป็นตัวตัดสิน account
2. hash token แล้วหา row ที่ยังไม่ใช้และไม่หมดอายุ
3. ตรวจ password policy: password ใหม่อย่างน้อย 12 ตัวอักษร หรือ passphrase ที่ผ่าน policy; ไม่เก็บ plaintext
4. transaction: mark token used, update password/passwordChangedAt, clear lockout, revoke all sessions, create audit
5. enqueue `PASSWORD_CHANGED` security email
6. response generic และ redirect ไป sign-in

### Security rules

- ห้าม auto-login หลัง reset; ให้ sign in ใหม่ผ่าน session ที่เพิ่งสร้าง
- password reset ไม่ bypass email verification policy ถ้า email ยังไม่ verified
- invalidate all sessions รวม current session เมื่อ reset สำเร็จ
- `GET /auth/reset-password?token=...` ต้อง hash/เก็บ encrypted reset intent ใน HttpOnly short-lived cookie แล้ว redirect ไป clean URL; หน้า reset ใช้ `no-store`/`no-referrer`, ปิด analytics และ POST เท่านั้นที่ consume token
- ห้ามใส่ token ใน application log, error message, query analytics หรือ audit metadata
- bcrypt cost ให้ benchmark ใน production-like environment; ค่าเริ่มต้นแนะนำ 12 หาก latency ยอมรับได้

### Acceptance criteria

- unknown email และ known email ได้ status/body shape เดียวกัน
- reset token ใช้ซ้ำไม่ได้ แม้มี concurrent requests
- reset สำเร็จแล้ว session เดิมทุกตัวใช้ไม่ได้
- password เก่าล็อกอินไม่ได้และ password ใหม่ใช้ได้
- token/password ไม่ปรากฏใน logs, audit หรือ response
- reset link ไม่ค้างใน URL/history/referrer และ mail scanner ไม่ consume token

### Tests

- generic forgot response
- token hash, expiry, replay และ race
- password policy/bcrypt
- session revocation
- rate limit and mail-provider failure

---

## 12. System Plan 8: PDPA/Data Privacy Lifecycle

### เป้าหมาย

ให้ผู้ใช้จัดการข้อมูลได้จริง และให้ระบบลบ/anonymize ข้อมูลตาม purpose โดยไม่ลบหลักฐานทางบัญชีหรือ security audit ที่ต้อง retain โดยไม่จำเป็น

### Files ที่เกี่ยวข้อง

- Modify: `prisma/schema.prisma`
- Create: migration for `DataRequest`, `UserConsent`, nullable financial/user relations and `PaymentRecord.userId` SetNull behavior
- Modify: `app/api/account/export/route.ts`
- Modify: `app/api/account/route.ts` deletion flow
- Create: `app/api/account/data-requests/route.ts`
- Create: `lib/data-lifecycle.ts`
- Create: `lib/retention.ts`
- Modify: backend cleanup/file lifecycle endpoint
- Create: scheduled cleanup worker/cron route
- Modify: privacy policy documentation
- Test: `tests/data-privacy.test.ts`

### Data inventory

| Data | Purpose | Default retention | Delete/anonymize action |
|---|---|---:|---|
| User profile | account/service delivery | จนกว่าผู้ใช้ลบ | delete หรือ anonymize |
| OAuth account tokens | login/provider connection | จน unlink/ลบ account | revoke provider token ถ้าทำได้ แล้ว delete |
| Session | authentication | จนหมดอายุ/revoke + 30 วัน | delete |
| Verification/reset tokens | one-time auth | 24 ชั่วโมงหลัง expiry/use | hard delete |
| ProjectRecord | history/UX | 90 วันหรือ policy ปัจจุบัน | delete พร้อม file cleanup |
| Temporary audio | processing | `SEPARATE_TTL_SECONDS` | backend cleanup ทันทีเมื่อครบ TTL |
| Audit security | fraud/support/security | 365 วัน | anonymize เมื่อ user purged |
| Payment/accounting | accounting/legal | 7 ปีหรือนโยบายกฎหมายที่อนุมัติ | retain minimal, detach PII |
| Notification delivery | delivery/security evidence | 90 วัน | delete/sanitize payload |
| Consent | legal evidence | ตาม policy version + legal period | retain evidence, record revoke |

### Export flow

- เปลี่ยนจาก synchronous JSON อย่างเดียวเป็น `DataRequest` ที่มี status และ expiry เมื่อข้อมูลเกิน response size ที่ปลอดภัย
- `POST /api/account/data-requests` สร้าง export request แบบ idempotent ต่อ user ต่อช่วงเวลาสั้น ๆ
- worker รวบรวม profile, preferences, subscription summary, quotas, project history, connected providers, consent และ user-visible activity
- ไม่ export password/hash, OAuth token, session token, raw IP/hash, internal audit metadata, Omise secrets หรือ audio bytes โดย default
- เก็บ export ใน private storage พร้อม one-time download token/authorization; expiry 24 ชั่วโมง แล้วลบทิ้ง
- audit request/completion/download

### Delete flow

1. user re-auth ด้วย password หรือ recent OAuth re-auth
2. transaction เปลี่ยน status เป็น `PENDING_DELETION`, ตั้ง grace period 30 วัน, revoke sessions และปิด login/backend processing ทันที; เปลี่ยน `ProcessingJob.phase` ที่ยังไม่ terminal เป็น `CANCELED` ด้วย claim/fence token และ invalidate file grants/backend upload intents
3. เปลี่ยน pending/reset/email-change tokens เป็น used/revoked, mark `QuotaReservation` ที่ปลอดภัยเป็น `REFUNDED` และคง `UNKNOWN/RECONCILING` จน reconcile เสร็จ
4. สร้าง cancellation task ของ Omise; `destroy` แบบ best effort เป็นแค่ attempt แรกและต้อง retry/ตรวจ provider state จน confirmed หรือ manual override ที่มี audit เหตุผล ก่อน anonymize/purge user
5. worker สร้าง `ArtifactCleanup`/export deletion manifest แล้วลบ project files ผ่าน internal authenticated backend cleanup; artifact ที่ cleanup สำเร็จลบได้ ส่วน artifact ที่ยังรอ/ล้มเหลวต้องเปลี่ยนเป็น `ownerType = PURGED`, ตัด `userId/ownerKeyHash`, ตั้ง `purgeDeadline` และปิด access ก่อนลบ/anonymize user
6. หลัง grace period และ cancellation/ownership gates ผ่าน:
   - ลบ profile, OAuth links, sessions, preferences และ non-required data เมื่อไม่มี `AudioArtifact.userId` ที่ยังผูกอยู่; `onDelete: Restrict` ต้องทำให้ cleanup worker แก้ owner state ก่อนเสมอ
   - anonymize audit เป็น `userId = null`, ลบ metadata ที่ระบุตัวบุคคล
   - detach/anonymize payment records ที่ต้อง retain; ห้ามลบ card data เพราะระบบไม่ควรเก็บอยู่แล้ว
   - เปลี่ยน status เป็น `ANONYMIZED`
7. account ที่ `PENDING_DELETION` สามารถ restore ได้ผ่าน explicit recovery flow ภายใน grace period; หลัง anonymize restore ไม่ได้ และ `PURGED` artifact จะถูกลบเมื่อ `ArtifactCleanup` สำเร็จ

### Schema/lifecycle invariants

- `PaymentRecord.userId` ต้องเปลี่ยนเป็น optional และ relation ใช้ `onDelete: SetNull` ใน migration แยก; เก็บเฉพาะ amount/currency/status/provider charge reference ที่จำเป็นต่อบัญชี และไม่เก็บ card data
- `User.email` ยังคง required/unique สำหรับ active users แต่ตอน anonymize ต้องแทนด้วย deterministic tombstone เช่น `deleted_${userId}@anonymized.invalid`, ตั้ง `name/image/password/omiseCustomerId` เป็น `null` และตั้ง `status = ANONYMIZED`
- `Subscription`/`UsageQuota`/`ProjectRecord` ต้องกำหนดทีละ model ว่าลบ, detach หรือเก็บ aggregate ก่อนเขียน migration; ห้ามอาศัย cascade เดิมโดยไม่ทดสอบผลต่อ financial/audit retention
- account purge ห้ามจบจาก Omise `destroy` attempt อย่างเดียว: ต้องมี cancellation task/status ที่ confirmed หรือมี manual override พร้อม actor/reason/audit; provider outage ทำให้ account ค้าง `PENDING_DELETION` และ retry ได้ ไม่ใช่ anonymize แล้วปล่อย schedule ทำงานต่อ
- `AuditLog`, `UserConsent`, `NotificationJob` ต้อง anonymize field-by-field (`userId`, actor/target user IDs, user-agent, IP hash, encrypted recipient/payload) และยกเลิก pending notification ที่ส่งต่อไม่ได้เมื่อ account ถูก purge
- `QuotaReservation` ที่เป็น `UNKNOWN/RECONCILING` ห้ามถูก cascade ตอนลบบัญชี; ต้อง reconcile backend outcome ก่อน แล้วจึง refund/consume/delete reservation หรือคงไว้ตาม financial audit policy
- Next.js เป็น metadata/control-plane owner: สร้าง `AudioArtifact`/`ProcessingJob`/`ArtifactCleanup`, ต่ออายุ lease จาก signed status callback ของ FastAPI และออก cleanup manifest ที่มี `claimToken`/fencing token; FastAPI เป็น data-plane owner ของ disk และยอมลบเฉพาะ artifact ID ใน manifest ที่ตรวจ service signature แล้ว
- cleanup ปกติทำเฉพาะ `AudioArtifact` ที่ terminal state (`COMPLETED`, `FAILED`, `CANCELED`) และ `expiresAt < now`; `PURGED` artifact ใช้ `purgeDeadline` แยกจาก normal TTL ได้หลัง backend job ถูก fence/cancel และไม่เปิด owner access แม้ cleanup ยัง pending
- เมื่อถึง `purgeDeadline` ต้องมี signed/fenced cancellation acknowledgement หรือเก็บ artifact เป็น tombstone แบบ ownerless; ห้ามลบ bytes ของงานที่ backend ยังอาจเขียนอยู่เพียงเพราะ TTL หมด
- FastAPI ส่ง cleanup result/attempt กลับ Next.js แบบ idempotent โดยต้อง match `claimToken`; worker crash/restart reclaim `ArtifactCleanup.leaseUntil` พร้อม token ใหม่, และ orphan reconciliation ค้นไฟล์ที่ไม่มี artifact/manifest แล้ว quarantine ก่อนลบผ่าน retry/dead-letter path
- worker ต้องลบ private export object ตาม `ExportDeletion.storageKey` โดย match `claimToken` ก่อน mark request purged และไม่ปล่อย download token ค้าง; การลบ DataRequest row ห้ามลบ ExportDeletion manifest ก่อน object deletion สำเร็จ

### Acceptance criteria

- export ไม่เปิด secret หรือ PII ที่ไม่จำเป็น
- delete ปิดการใช้งานทันที แต่ cleanup ทำซ้ำได้และ recover จาก partial failure
- file cleanup มี owner authorization/internal service token และไม่ลบ file ของ user อื่น
- audit/financial retention ไม่ถูก cascade ลบโดยไม่ตั้งใจ
- retention job มี metric, retry และ dead-letter/error report
- cleanup ไม่ลบงานที่ยังประมวลผล, ทำซ้ำได้เมื่อ worker/restart และตรวจครบ `uploads`, `separated`, `eq_applied`, `compressed`

### Tests

- export ownership and redaction
- idempotent data request
- delete transaction + grace period + restore
- partial backend cleanup retry
- anonymization removes direct PII but keeps required event/accounting fields
- retention job deletes only records past `retentionUntil`
- active-job lease/heartbeat and orphan-file reconciliation
- purge cancels/fences issued jobs and backend intents before owner removal; `PURGED` artifacts have a forced `purgeDeadline`
- unknown quota reservations survive account purge until fenced reconciliation completes
- Omise cancellation must be confirmed/retried before final anonymization unless audited manual override

---

## 13. System Plan 9: Notification & Email Delivery

### เป้าหมาย

ส่ง email ที่เกี่ยวกับ security, verification, billing และ product โดยไม่ผูก business route กับ provider และไม่ทำให้ request หลักล้มเพราะ provider ช้า/ล่ม

### Files ที่เกี่ยวข้อง

- Create: `lib/email.ts`
- Create: `lib/notification.ts`
- Create: `lib/email-templates.ts`
- Modify: `prisma/schema.prisma` (`NotificationJob`)
- Create: migration
- Create: `app/api/internal/notifications/worker/route.ts` หรือ worker process ตาม deployment
- Modify: register, verification, reset, session, billing and deletion flows
- Create: provider adapter tests and integration tests

### Notification classes

| Type | Opt-out ได้หรือไม่ | ตัวอย่าง |
|---|---|---|
| Essential security | ไม่ได้ | verification, password reset, password changed, new login, session revoked |
| Essential service/billing | ไม่ได้ถ้าจำเป็นต่อบริการ | payment receipt, payment failed, subscription status |
| Product | ได้ | processing complete, quota warning |
| Marketing | ต้องมี explicit consent | promotion, newsletter |

### Outbox flow

```text
Business transaction
  -> create NotificationJob with dedupeKey
  -> worker claims PENDING job atomically
  -> provider adapter sends email
  -> mark SENT or retry with backoff
  -> exhausted job becomes FAILED + alert
```

`NotificationJob` ต้องเก็บ `recipientCiphertext` และ template variables ที่จำเป็นใน `payloadCiphertext` แบบเข้ารหัส เพื่อให้ worker retry ได้โดยไม่ต้องเก็บ raw token/email; worker decrypt เฉพาะใน memory และไม่ log ค่าเหล่านี้ เมื่อ user ถูกลบให้ cancel job ที่ยังไม่ส่งแทนการพยายามส่งหลัง anonymize

การ claim ใช้ conditional update จาก `PENDING/RETRY` หรือ `PROCESSING` ที่ `leaseUntil < now` ไป `PROCESSING`, ตั้ง `claimedBy`/`leaseUntil`; worker crash จึงถูก reclaim ได้ และ `nextAttemptAt` เป็น source สำหรับ scheduler

### Provider interface

```typescript
type EmailMessage = {
  to: string;
  template: "VERIFY_EMAIL" | "PASSWORD_RESET" | "PASSWORD_CHANGED" | "NEW_LOGIN" | "PAYMENT_RECEIPT";
  variables: Record<string, string>;
  dedupeKey: string;
  idempotencyKey: string;
};

interface EmailProvider {
  send(message: EmailMessage): Promise<{ providerMessageId?: string }>;
}
```

Templates ต้อง escape user-controlled name/file text, ไม่ใส่ secret/token ใน subject และ token link ต้องมี expiry/HTTPS/base URL จาก server config เท่านั้น

### Retry policy

- retry network/5xx ด้วย exponential backoff สูงสุด 5 ครั้ง
- 4xx invalid recipient/template เป็น permanent failure; ไม่ retry ไม่จำกัด
- dedupe ด้วย `dedupeKey` เช่น `password-changed:{userId}:{passwordChangedAt}`
- `providerIdempotencyKey` ต้องคงเดิมทุก retry และส่งเป็น provider idempotency header/key; provider adapter ที่ไม่มี dedupe capability ห้ามใช้กับ essential security email ใน production
- worker claim ต้อง atomic เพื่อไม่ส่งซ้ำจากหลาย instance และ completion ต้องเป็น fenced update ที่ตรวจ `claimedBy` + lease token ก่อน mark `SENT`
- notification failure ไม่ rollback password/security state แต่ต้อง alert และสร้าง audit/operational event

### Acceptance criteria

- password reset/change state สำเร็จได้แม้ email provider ช้า แต่ user เห็น generic response
- job ไม่ส่งซ้ำจาก concurrent workers
- user opt-out product notification แล้วไม่หยุด essential security email
- provider error ไม่รั่ว token/email body ใน logs
- delivery status ค้นได้โดย admin ที่ได้รับอนุญาต
- worker crash/restart reclaim job ได้ และ account purge ไม่ทิ้ง job ที่จะส่งข้อมูลต่อ
- provider รับคำขอแล้ว worker crash/retry ใช้ idempotency key เดิมและไม่สร้าง email ซ้ำใน provider ที่รองรับ

### Tests

- outbox creation transaction
- dedupe/concurrent claim
- retry/backoff/permanent failure
- provider idempotency and fenced completion after worker crash
- opt-in/opt-out policy
- template escaping and absence of secrets

---

## 14. System Plan 10: Admin, RBAC & Support Operations

### เป้าหมาย

ให้ทีม support/owner ตรวจสอบและแก้ปัญหา user ได้โดยไม่ใช้ Prisma Studio หรือแก้ database ตรง และทุก privileged action มีหลักฐานตรวจสอบย้อนหลัง

### Files ที่เกี่ยวข้อง

- Modify: `prisma/schema.prisma` (`User.role`, `User.status`)
- Create: `lib/admin.ts`
- Create: `lib/admin-mfa.ts`
- Create: `lib/permissions.ts`
- Create: `app/api/admin/step-up/route.ts`
- Create: `app/api/admin/users/route.ts`
- Create: `app/api/admin/users/[id]/route.ts`
- Create: `app/api/admin/users/[id]/sessions/route.ts`
- Create: `app/api/admin/audit/route.ts`
- Create: `app/api/admin/notifications/route.ts`
- Create: `app/admin/page.tsx` และ components หลังอ่าน `DESIGN.md`
- Test: `tests/admin.test.ts`

### Roles and permissions

| Permission | USER | SUPPORT | ADMIN |
|---|---:|---:|---:|
| ดู account ของตนเอง | yes | yes | yes |
| ค้น user แบบ masked | no | yes | yes |
| ดู user activity | own | masked/support scope | all |
| revoke session | own | yes with reason | yes |
| suspend/unsuspend | no | no | yes |
| เปลี่ยน tier | no | no | yes with reason |
| ดู audit/security | own filtered | limited | all |
| จัดการ role | no | no | separate break-glass only |

เริ่มต้นใช้ `User.role` แบบสามระดับก่อน ไม่สร้าง dynamic permission matrix จนกว่าจะมี requirement multi-tenant หรือหลายทีมจริง `SUPPORT` ใน MVP ไม่มี assignment model และเห็นเฉพาะ masked PII/limited activity; ถ้าต้องแบ่งเคสรายทีมให้เพิ่ม `SupportAssignment` เป็น workstream แยก ไม่ใช้ filter จาก client

ADMIN ต้องใช้ step-up security ก่อนเปิด production: ค่าเริ่มต้นให้ผูกกับ SSO/VPN ที่บังคับ MFA; ถ้าใช้ credentials ในแอปต้องเพิ่ม TOTP/WebAuthn และห้ามเปิด admin mutation จน factor ผ่าน การ bootstrap admin ใช้ one-time invite หรือ secret manager ที่ไม่พิมพ์ password ลง `prisma/seed.ts`, บังคับเปลี่ยน credential และ audit การสร้าง/เปลี่ยน role

### Admin API rules

- ทุก route เรียก `requireAdminSession` และ `requirePermission`
- `requireAdminSession` ต้องตรวจ database session ปัจจุบัน, `User.role/status`, `Session.adminMfaVerifiedAt > now - 10 นาที` และ `adminMfaMethod` ที่อยู่ใน allowlist; `lastAuthenticatedAt` อย่างเดียวไม่ถือเป็น MFA
- `POST /api/admin/step-up` ต้องผูกการยืนยันกับ session ID ปัจจุบัน, nonce/challenge และ method (SSO auth event, TOTP หรือ WebAuthn); ห้ามรับ boolean/role/MFA timestamp จาก client
- role change, suspend, password reset, session revoke-all และ admin sign-out ต้อง clear `adminMfaVerifiedAt` ของ session ที่เกี่ยวข้อง และ role/status change ต้อง revoke admin sessions อื่น
- mutation ต้องใช้ CSRF/origin check, re-auth สำหรับ action สำคัญ และรับ `reason` ที่มีความยาว/รูปแบบจำกัด
- query user ต้อง mask email เป็นค่าเริ่มต้น; เปิด full PII เฉพาะ permission ที่จำเป็น
- ห้ามส่ง password, OAuth token, session token, raw IP/hash หรือ payment secret
- query ต้องมี index/limit สำหรับ `User.status`, `User.role`, `User.lastLoginAt`, normalized email และห้ามใช้ email/IP เป็น high-cardinality metric label
- admin ไม่ควรแก้ `PaymentRecord` หรือ Omise state โดยตรง; ใช้ service ที่ idempotent และบันทึกเหตุผล
- ทุก read ของ user-sensitive data และทุก mutation สร้าง audit event พร้อม actor/target/reason
- ป้องกัน admin self-demotion/last-admin lockout ด้วย transaction และ break-glass procedure นอก public API

### Dashboard MVP

- user search/filter: id hash, masked email, status, role, tier, created date, last login
- user detail: subscription/quota summary, recent user-visible activity, sessions metadata, verification/status
- actions: suspend/unsuspend, revoke sessions, resend verification, trigger password reset email
- audit explorer: event/outcome/date/actor/target/request ID พร้อม pagination
- operational widgets: auth failure, lockout, 429, email failure, backend token rejection, audit write failure

### Acceptance criteria

- USER เรียก admin API ไม่ได้แม้ปลอม role ใน session/client
- SUPPORT ทำ action ที่ไม่มี permission ไม่ได้
- ADMIN action ทุกครั้งมี actor, target, reason และ audit event
- ADMIN API ปฏิเสธ session ที่ไม่มี required MFA/step-up state และไม่มี production path ที่พิมพ์ password จาก seed
- list/query มี pagination, maximum page size และไม่มี unbounded database query
- admin UI ไม่แสดง secret และรองรับ mobile/desktop ตาม `DESIGN.md`

### Tests

- role/permission matrix
- IDOR: admin API cannot access arbitrary internal fields without permission
- CSRF/re-auth/reason validation
- admin step-up is session-bound, expires after 10 minutes, and is cleared on role/status change
- TOTP/WebAuthn/SSO factor verification updates only the current session and stores encrypted factor material
- last-admin protection
- audit emitted for read and mutation
- pagination/filter injection and rate limits

---

## 15. System Plan 11: CSRF, Security Headers & Request Integrity

### เป้าหมาย

ป้องกัน cookie-authenticated mutation, clickjacking, MIME sniffing, unsafe referrer และ configuration ที่ทำให้ browser trust application มากเกินไป

### Files ที่เกี่ยวข้อง

- Create: `lib/security/csrf.ts`
- Create: `lib/security/origin.ts`
- Modify: `middleware.ts`
- Modify: `next.config.ts` หรือ root security headers configuration
- Modify: `app/api/account/*`, auth mutation, subscription, history delete, quota routes
- Modify: `backend/main.py` CORS/host middleware
- Test: `tests/security-headers.test.ts`
- Test: `tests/csrf.test.ts`
- Test: `backend/tests/test_security_middleware.py`

### Policy

- Production cookie: `Secure`, `HttpOnly`, `SameSite=Lax` หรือ `Strict` ตาม OAuth callback compatibility
- Unsafe methods `POST`, `PUT`, `PATCH`, `DELETE` ที่ใช้ session cookie ต้องตรวจ `Origin` ให้ตรงกับ allowlist และ reject missing/foreign origin
- custom browser mutation ทุก route ต้องใช้ double-submit CSRF token ร่วมกับ origin check; มีข้อยกเว้นเฉพาะ NextAuth built-in endpoints และ webhook/service endpoints ที่ไม่มี cookie auth พร้อม route-level test matrix
- CSRF token ห้ามอยู่ใน log, audit metadata, analytics หรือ error response
- NextAuth endpoints ใช้ built-in CSRF และต้องไม่ถูกครอบด้วย custom rule ที่ทำให้ OAuth callback พัง
- Omise webhook ใช้ HMAC/event verification ไม่ใช้ cookie/CSRF
- backend direct service endpoint ใช้ bearer token ไม่รับ cookie auth

Omise webhook ต้องบันทึก `WebhookEvent(provider, providerEventId)` แบบ unique ก่อนประมวลผล; row ใหม่/row ที่ lease หมดอายุเท่านั้นจึง claim ไป `PROCESSING`, ทำ business mutation และ mark `PROCESSED` ใน transaction/idempotent service เดียวกัน เมื่อ provider รองรับให้ส่ง idempotency key เดิมตอน retry หาก worker ตายหลัง provider รับ event แล้ว duplicate จะถูก dedupe ด้วย event ID/charge ID ไม่สร้าง mutation ซ้ำ

ตรวจ event timestamp/order และยอมรับเฉพาะ state transition ที่ไม่ downgrade เช่น `successful` แล้วห้ามเปลี่ยนกลับเป็น `pending`; `metadata.userId/tier` จาก Omise ใช้เป็น hint สำหรับค้นหาเท่านั้น ต้อง match กับ server-created checkout intent/customer/schedule/charge record ก่อนเปลี่ยน subscription

### Headers

Production baseline:

- `Strict-Transport-Security` เมื่อ HTTPS ครบทุก subdomain ที่เกี่ยวข้อง
- `Content-Security-Policy` เริ่ม `Report-Only`, เก็บ violation แล้วค่อย enforce
- `X-Content-Type-Options: nosniff`
- `Referrer-Policy: strict-origin-when-cross-origin`; reset/password pages ใช้ `no-referrer`
- `Permissions-Policy` ปิด feature ที่ไม่ใช้
- `frame-ancestors 'none'` ผ่าน CSP แบบ enforce ตั้งแต่เริ่ม; CSP script/connect policy จะเริ่ม Report-Only แล้วค่อย enforce หลังแก้ violation
- `Cache-Control: no-store` สำหรับ account/security/API responses ที่มี PII

FastAPI:

- `ALLOWED_ORIGINS` เป็น explicit production origins ไม่มี wildcard
- จำกัด methods/headers เหลือที่ใช้จริงหลัง backend token migration
- เพิ่ม trusted host/proxy configuration และไม่ trust `X-Forwarded-For` จาก public client
- เพิ่ม upload timeout/body limit และ response header ที่จำเป็นเท่านั้น

### Acceptance criteria

- cross-origin POST/DELETE ที่ใช้ cookie ถูก reject
- same-origin account/payment mutation ทำงานปกติ
- webhook ไม่ถูก block ด้วย CSRF แต่ signature ไม่ถูกต้องถูก reject
- webhook event เดิมถูก dedupe ด้วย provider event ID และไม่ทำ billing mutation ซ้ำ
- webhook events for multiple resources use `(provider, resourceKey)` ordering markers and reject stale resource versions
- concurrent webhook delivery uses row lock/serializable/CAS and missing ordering fields fail closed without billing mutation
- response headers ผ่าน automated security header test
- CSP violations ไม่มี secret/PII ใน report payload และมี monitoring

---

## 16. Metrics, Alerts & Operations

Audit log ไม่แทน metrics และ operational log ไม่แทน audit evidence ต้องมีสามชั้นแยกกัน:

| ชั้น | ใช้ตอบคำถาม | ที่เก็บ |
|---|---|---|
| Metrics | ระบบช้าหรือเสียแค่ไหน | Prometheus/hosted metrics |
| Operational logs | request/job ใดผิดพลาด | JSON stdout/log collector |
| Audit logs | ใครเปลี่ยน account/security/business state | Prisma `AuditLog` |

### Metrics ขั้นต่ำ

- `auth_login_success_total`, `auth_login_failure_total`
- `auth_lockout_total`, `auth_rate_limited_total`
- `email_verification_sent_total`, `password_reset_requested_total`, `email_delivery_failed_total`
- `audit_write_failure_total`
- `backend_token_rejected_total`
- `csrf_rejected_total`
- `audio_job_started_total`, `audio_job_failed_total`, `quota_refund_total`
- `active_sessions`, `notification_queue_depth`, `data_request_queue_depth`
- `http_request_duration_ms`, `http_5xx_total`

### Alert defaults

- audit write failure มากกว่า 0 ต่อเนื่อง 5 นาที
- Redis unavailable มากกว่า 60 วินาที
- backend token rejection พุ่งผิดปกติจาก baseline หรือมีหลาย issuer/audience failure
- auth failure/lockout หรือ reset request spike จาก IP/region เดียว
- email delivery failure มากกว่า 5% ใน 15 นาที
- notification/data-request queue ค้างเกิน SLA
- 5xx สูงกว่า 2% ใน 5 นาที
- cleanup failure ทำให้ temporary file/disk usage เกิน threshold

### Runbooks ที่ต้องมี

- rotate `NEXTAUTH_SECRET`, backend signing key, PII HMAC key, email provider key
- Redis outage และ fallback mode
- suspected credential stuffing/account takeover
- revoke all sessions ของ user/ทุก user
- stuck notification/data deletion job
- restore database และตรวจ audit continuity
- disk pressure จาก audio output/cleanup failure

---

## 17. Implementation Roadmap

แต่ละ phase ต้อง merge ได้และมี verification ของตัวเอง ไม่รวม migration ใหญ่ที่ไม่มี rollback หรือ monitoring plan

### Phase 0: Production prerequisites

- [ ] ตัดสินใจ deployment topology: single instance, multi-instance หรือ serverless
- [ ] สร้าง PostgreSQL staging และ migration path จาก SQLite
- [ ] สร้าง Redis/Upstash staging และ secret manager entries
- [ ] เลือก email provider และ verify domain/DKIM/DMARC
- [ ] ตั้ง `NEXTAUTH_SECRET`, `BACKEND_AUTH_SECRET`/key set, `PII_HASH_SECRET`, `APP_ENCRYPTION_KEY`/KMS key และ webhook secret แยกกัน
- [ ] ตั้ง log collector, metrics, alert destination และ backup/restore test
- [ ] กำหนด privacy/retention policy version สำหรับ `/privacy`

PostgreSQL migration ต้องมี runbook ที่ทำซ้ำได้: backup SQLite, ตรวจ/normalize duplicate email, export/import ด้วย script ที่นับ rows และ checksum ของ key สำคัญ, run Prisma migrations ใน staging, compare User/Subscription/Payment/Project counts, smoke-test auth/billing/quota, เปิด read-only maintenance window, snapshot ก่อน cutover และกำหนด abort criteria จาก migration error/row mismatch/latency. Rollback หลังเริ่ม writes ให้เป็น restore/switch ไป pre-cutover snapshot หรือ forward-fix ที่อนุมัติแล้ว ไม่ใช่สั่งย้อน migration แบบ blind เพราะสองฐานข้อมูลจะ diverge

Prisma migration ต้องใช้ expand/contract: เพิ่ม nullable/index/model ก่อน, backfill/verify, เปิด code path ใหม่, แล้วค่อยบังคับ constraint ใน migration ถัดไป. ส่วน PostgreSQL functional lower-email index, `CHECK` XOR owner, `SetNull` financial relation และ locking/index behavior ที่ SQLite ไม่รองรับตรงกันต้องมี provider-specific SQL migration พร้อม CI matrix รันทั้ง SQLite และ PostgreSQL; ห้ามใช้ schema เดียวเป็นเหตุผลให้ละเลย constraint ฝั่ง production

Rollback release ต้องเป็น security-compatible release เท่านั้น: legacy code ที่อ่าน `X-User-*`, quota fail-open หรือ public file endpoint ห้ามถูกเปิดกลับมาโดยอัตโนมัติ หากต้องย้อนกลับก่อน migration เสร็จให้ปิด traffic/เข้า maintenance และ deploy compatibility release ที่ยัง reject legacy authority; ห้ามตั้ง flag เพื่อ downgrade security เพียงเพื่อให้ old UI ทำงาน

### Phase 1: Security foundation (P0)

- [ ] Structured logger, redaction, request ID สำหรับ Next.js/FastAPI
- [ ] `AuditLog` model + audit helper + events สำหรับ auth/account/billing/quota
- [ ] Redis rate limit + auth abuse/lockout
- [ ] Quota reservation/operation ID และ guest quota shared-store migration
- [ ] CSRF/origin checks + security headers + trusted proxy handling
- [ ] Backend signed token/BFF และลบ trust จาก `X-User-*`
- [ ] Artifact ownership/file-grant matrix และ webhook event dedupe
- [ ] OAuth token encryption compatibility release, backfill/verify, key rotation และ plaintext-backup retirement
- [ ] เพิ่ม metrics/alerts ของ audit, rate limit, token rejection และ 5xx

Phase 1 backend-token rollout ยังไม่เปิด email-verification enforcement เอง: `REQUIRE_EMAIL_VERIFICATION_FOR_PROCESSING=false` จนกว่า Phase 2 จะทำ legacy backfill, campaign และ verify grace-period behavior เสร็จ; ระหว่างนั้น BFF ตรวจ `emailVerificationRequiredAt` ตาม policy ที่ versioned ไม่ใช้ status default เดาเอง

OAuth encryption cutover ใช้ compatibility release ที่อ่าน plaintext เก่าได้เฉพาะ migration worker แต่เขียน ciphertext ใหม่, ตรวจ count/hash/decrypt ของทุก Account, deploy release ที่ปิด plaintext read, rotate/re-encrypt ด้วย key version ใหม่ และ retire encrypted backups/old key ตาม retention หลัง verify. Rollback หลังปิด plaintext compatibility ใช้ encrypted snapshot + key version ที่ยัง active หรือ forward-fix เท่านั้น; ห้ามเปิด plaintext token read กลับใน production หลัง cutover สำเร็จ

### Phase 2: Auth lifecycle (P1)

- [ ] Email verification/email change
- [ ] Legacy-user email/status backfill และ canonical-email uniqueness migration
- [ ] Password reset/recovery
- [ ] Notification outbox สำหรับ security email
- [ ] เปลี่ยนเป็น database session และ session/device management
- [ ] Re-auth ย้ายจาก client-visible `reauthAt` ไป server-side session state

Phase 2 ต้องเปิด `REQUIRE_EMAIL_VERIFICATION_FOR_PROCESSING=true` หลัง backfill/campaign verification และ monitoring ผ่านเท่านั้น; ถ้า rollback ให้ปิด feature ใหม่ด้วย compatibility release ที่ยังไม่เปิด legacy header/file/quota fail-open

### Phase 3: Privacy and support operations (P1/P2)

- [ ] DataRequest async export + redaction
- [ ] Deletion grace period, backend file cleanup, anonymization, retention worker
- [ ] `ProcessingJob` lease/heartbeat/orphan reconciliation และ PaymentRecord detach migration
- [ ] UserConsent records และ update privacy/terms flows
- [ ] Admin RBAC/API + audit explorer
- [ ] Admin dashboard และ support runbooks
- [ ] Admin MFA/SSO/VPN bootstrap และ production seed credential removal

### Phase 4: Hardening and scale

- [ ] Load test login/reset/rate limit/audio BFF
- [ ] Chaos test Redis/email/backend/database failure
- [ ] PostgreSQL indexes/query review และ audit retention partition/archive ตาม volume
- [ ] CSP enforce หลัง report-only period
- [ ] Penetration test/OWASP ASVS review
- [ ] Disaster recovery restore drill และ incident response tabletop

---

## 18. Workstream Dependency Order

```text
Production secrets/database/Redis
          |
          v
Request context + redaction + structured logs
          |
          +--> AuditLog helper/events
          |
          +--> Redis rate limit + lockout
          |
          +--> CSRF/security headers
          |
          v
Backend signed token/BFF
          |
          +--> Email provider/outbox
          |       |
          |       +--> Email verification
          |       +--> Password reset
          |
          v
Database sessions/device management
          |
          v
Data export/delete/retention
          |
          v
Admin RBAC/dashboard
```

ห้ามเริ่ม admin dashboard ก่อน permission/audit foundation และห้ามเปิด backend token migration ก่อนมี request ID/logging เพราะจะวิเคราะห์ปัญหา trust boundary ได้ยาก

---

## 19. Verification Checklist ก่อนประกาศ Production-ready

### Functional

- [ ] Register -> verify email -> login -> audio process ทำงานครบ
- [ ] Forgot/reset password ใช้ one-time token และ revoke sessions
- [ ] User ดู/revoke session ได้ และ revoke all มีผลจริง
- [ ] User export/delete ทำงานตาม retention/grace period
- [ ] Admin search/action ทำได้เฉพาะ role ที่ถูกต้อง
- [ ] Backend รับเฉพาะ verified token หรือ guest flow ที่กำหนด
- [ ] User/guest download, playback, export ตรวจ owner/file grant และไม่มี IDOR
- [ ] Quota reserve/consume/refund ผูก operation ID และทำซ้ำแล้วไม่เปลี่ยนยอดผิด
- [ ] Account deletion ไม่ลบ user ก่อน artifact/export cleanup; unresolved artifacts กลายเป็น `PURGED` โดยไม่มี owner access

### Security

- [ ] ไม่มี route ไหนใช้ `X-User-ID`/`X-User-Tier` เป็น authority
- [ ] ไม่มี secret/PII ตามรายการต้องห้ามใน operational log/audit metadata
- [ ] Login/reset/register มี distributed rate limit และ abuse test
- [ ] Cross-origin mutation และ invalid CSRF ถูก reject
- [ ] session revoke, password reset, account delete revoke access ตาม policy
- [ ] key rotation และ secret rotation ผ่าน runbook จริง
- [ ] signed-token fallback มี replay protection หรือถูกปิดเมื่อใช้ BFF-only
- [ ] webhook/backend token verification เป็น fail-closed
- [ ] webhook ordering/resource version กัน event เก่าดาวน์เกรด subscription
- [ ] verification/reset/email-change GET ล้าง token จาก URL และใช้ no-store/no-referrer/analytics suppression
- [ ] OAuth token ciphertext/key rotation ผ่าน migration verification และไม่มี plaintext backup ที่ยังใช้งาน

### Reliability

- [ ] Redis outage มี bounded fallback/alert
- [ ] Email provider outage ไม่ทำให้ password/security state เสีย แต่มี retry/alert
- [ ] Audit write failure มีสัญญาณเตือนและไม่มีข้อมูลลับหลุด
- [ ] cleanup/retention/data request job retry ได้และไม่ทำซ้ำผลเสีย
- [ ] notification worker มี lease/reclaim และ encrypted payload ไม่รั่ว token/email
- [ ] notification completion ตรวจ claim/fence token และ provider idempotency key
- [ ] PostgreSQL backup restore ผ่าน staging drill
- [ ] มี monitoring สำหรับ latency, 5xx, queue backlog, disk usage และ auth abuse

### Tests and quality

- [ ] `npm run type-check`
- [ ] `npm run lint`
- [ ] `npx jest`
- [ ] `pytest` หรือ `python -m unittest discover -s backend/tests`
- [ ] integration tests กับ PostgreSQL/Redis staging
- [ ] load/concurrency tests สำหรับ rate limit, quota, session revoke และ notification claim
- [ ] security review ตาม OWASP ASVS ที่เกี่ยวข้อง

---

## 20. Recommended First Implementation Slice

ถ้าจะเริ่มทำจริง ให้เริ่มจาก slice ที่ลดความเสี่ยงสูงสุดและไม่เปลี่ยน UX ใหญ่:

1. `lib/logger.ts` + request ID + redaction
2. `AuditLog` + `recordAuditEvent` สำหรับ login/register/account/password/subscription
3. Redis-backed rate limit และ login lockout
4. Quota reservation + artifact ownership/file-grant checks ก่อน backend token
5. Backend signed token validation โดยยังเปิด guest flow
6. CSRF/origin/security headers และ webhook dedupe

เมื่อ slice นี้ผ่าน test และ staging verification แล้วจึงทำ email verification/password reset/session management ต่อ เพราะทั้งสามระบบต้องใช้ audit, rate limit, notification และ session foundation ร่วมกัน

เอกสารนี้เป็น master roadmap สำหรับ production user management; ก่อนแก้โค้ดให้แตก Phase/Workstream ที่เลือกเป็น implementation plan แยกไฟล์ พร้อม file-level tasks, failing tests, migration rollback และ verification command ที่ตรงกับ scope นั้น
