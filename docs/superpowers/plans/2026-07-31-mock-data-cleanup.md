# Mock Data Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** กำจัด mock data, placeholder values, weak secrets, และ bug fix test assertion ทั่วโปรเจกต์ HarmoniQ

**Architecture:** แก้ไขไฟล์แยกอิสระ 8 tasks — ไม่มีการพึ่งพากันระหว่าง task สามารถทำพร้อมกันได้

**Tech Stack:** Next.js 15, React 19, TypeScript, Tailwind CSS 4, Prisma, FastAPI/Python 3.10

## Global Constraints

- เขียนคอมเมนต์ภาษาไทยตาม convention ของโปรเจกต์
- ห้ามใช้ Python 3.11+ features
- ห้ามรัน `npm run build` ระหว่าง dev — ใช้ `npx tsc --noEmit` แทน
- `.env` จริงต้องไม่อยู่ใน commit (มี `.gitignore` แล้ว)

---

### Task 1: ลบ MOCK_PAYMENTS ใน BillingSection

**Files:**
- Modify: `app/account/BillingSection.tsx:15-25`

**Interfaces:**
- Produces: `useState<PaymentRecord[]>([])` + existing loading/empty states

- [ ] **Step 1: ลบค่าคงที่ MOCK_PAYMENTS**

ลบ entire block (lines 15-21):
```ts
const MOCK_PAYMENTS: PaymentRecord[] = [
  { id: "mock-1", amount: 29900, currency: "thb", status: "successful", paidAt: "2026-07-25T08:30:00Z" },
  { id: "mock-2", amount: 29900, currency: "thb", status: "successful", paidAt: "2026-06-25T08:30:00Z" },
  { id: "mock-3", amount: 29900, currency: "thb", status: "successful", paidAt: "2026-05-25T08:30:00Z" },
  { id: "mock-4", amount: 9900, currency: "thb", status: "successful", paidAt: "2026-04-25T08:30:00Z" },
  { id: "mock-5", amount: 9900, currency: "thb", status: "failed", paidAt: "2026-04-20T10:15:00Z" },
];
```

- [ ] **Step 2: เปลี่ยน useState initial value**

เปลี่ยนบรรทัด 25 จาก:
```ts
const [payments, setPayments] = useState<PaymentRecord[]>(MOCK_PAYMENTS);
```
เป็น:
```ts
const [payments, setPayments] = useState<PaymentRecord[]>([]);
```

- [ ] **Step 3: ปรับ loading state text ให้อ่านง่ายขึ้น**

เปลี่ยนบรรทัด 136 จาก:
```tsx
<div className="text-[#8E8E8E] text-sm py-4">Loading payment history...</div>
```
เป็น:
```tsx
<div className="text-[#8E8E8E] text-sm py-4">กำลังโหลดประวัติการชำระเงิน...</div>
```

- [ ] **Step 4: ปรับ empty state text**

เปลี่ยนบรรทัด 140 จาก:
```tsx
<p className="text-[#888] text-sm">No payment records</p>
```
เป็น:
```tsx
<p className="text-[#888] text-sm">ยังไม่มีประวัติการชำระเงิน</p>
```

- [ ] **Step 5: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors related to BillingSection

- [ ] **Step 6: Commit**

```bash
git add app/account/BillingSection.tsx
git commit -m "fix: remove MOCK_PAYMENTS, use empty array initial state"
```

---

### Task 2: เปลี่ยน progress bar เป็น indeterminate ใน UploadBox

**Files:**
- Modify: `app/components/UploadBox.tsx:340-356, 845-879`

**Interfaces:**
- Consumes: existing `loading`, `progress`, `statusText`, `progressTimerRef` states
- Produces: indeterminate progress bar ที่ไม่แสดง % ปลอม

- [ ] **Step 1: ลบ progress simulation แบบ fake %**

เปลี่ยนบรรทัด 340-356 จาก:
```ts
// ทำ progress แบบจำลองไว้ก่อน เพราะ backend ไม่ได้ส่งสถานะระหว่างประมวลผลกลับมา
progressTimerRef.current = setInterval(() => {
  setProgress((prev) => {
    if (prev >= 90) return prev; // หยุดไว้ที่ 90% แล้วรอผลจริงจาก backend
    return prev + 2;
  });
}, 200);
```
เป็น:
```ts
// แสดง indeterminate progress animation ระหว่างรอ backend ประมวลผล
setProgress(0);
```

- [ ] **Step 2: ลบ stopProgressSimulation ที่เหลือ (ถ้ามี)**

Search ใน UploadBox.tsx ถ้ามี `setProgress(100)` หรือ `setProgress(0)` อื่นๆ ให้คงไว้ (ใช้รีเซ็ต state ได้)

แต่ต้อง remove `clearInterval(progressTimerRef.current)` ที่ถูกเรียกตอนจบ processing — เปลี่ยนเป็นแค่ `setProgress(0)` เฉยๆ

เช็คบริเวณ catch/finally block ของ handleSubmit (หาประมาณ line ~400-500) ว่ามี `clearInterval(progressTimerRef.current)` อยู่ ให้เปลี่ยนเป็น:
```ts
setProgress(0);
```
(ไม่ต้อง clear interval เพราะไม่มี interval แล้ว)

- [ ] **Step 3: เปลี่ยน UI progress bar เป็น indeterminate animation**

แทนที่ segmented meter bar (lines 845-879) ด้วย indeterminate bar:

```tsx
<div className="flex flex-col gap-1.5 w-full">
  {/* Indeterminate progress bar — เคลื่อนที่ไปมาไม่แสดง % */}
  <div className="w-full h-1.5 overflow-hidden rounded-full bg-[#111111] border border-[#1A1A1A]">
    <div 
      className={`h-full rounded-full transition-all ${
        action === "separate" ? "bg-gradient-to-r from-[#A78BFA] via-[#C084FC] to-[#A78BFA]" 
        : action === "eq-ai" ? "bg-gradient-to-r from-[#22D3EE] via-[#67e8f9] to-[#22D3EE]"
        : action === "compressor" ? "bg-gradient-to-r from-[#E5A93D] via-[#FBBF24] to-[#E5A93D]"
        : "bg-gradient-to-r from-[#34D399] via-[#6EE7B7] to-[#34D399]"
      } bg-[length:200%_auto] animate-shimmer`}
      style={{ width: "60%" }}
    />
  </div>
  <div className="flex justify-between items-center">
    <span className="text-[11px] text-[#555555]">
      {statusText || (loading ? "กำลังประมวลผล..." : "พร้อมใช้งาน")}
    </span>
    {/* ไม่แสดง % ปลอมอีกต่อไป */}
  </div>
  {processingTime && (
    <div className="text-[11px] text-[#444444] font-mono">⏱ {processingTime}</div>
  )}
</div>
```

- [ ] **Step 4: เช็คว่า `animate-shimmer` มีใน Tailwind config**

Run: `grep -r "shimmer" app/globals.css`
Expected: ✅ ได้รับการยืนยันแล้ว — `--animate-shimmer: shimmer 2s linear infinite` อยู่ที่ `app/globals.css:6` และ `@keyframes shimmer` อยู่ที่บรรทัด 19 แล้ว และ `animate-shimmer` ถูกใช้อยู่แล้วใน `UploadBox.tsx:859` → **ไม่ต้องแก้ globals.css**

- [ ] **Step 5: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add app/components/UploadBox.tsx
git commit -m "fix: replace fake progress % with indeterminate animation"
```

---

### Task 3: เปลี่ยน placeholder strings ใน auth.ts + omise.ts

**Files:**
- Modify: `lib/auth.ts:46-55`
- Modify: `lib/omise.ts:4-5`

- [ ] **Step 1: auth.ts — เปลี่ยน OAuth placeholder**

เปลี่ยนบรรทัด 46-47 จาก:
```ts
clientId: process.env.GOOGLE_CLIENT_ID || "google_placeholder",
clientSecret: process.env.GOOGLE_CLIENT_SECRET || "google_placeholder",
```
เป็น:
```ts
clientId: process.env.GOOGLE_CLIENT_ID || "",
clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
```

เปลี่ยนบรรทัด 50-51 จาก:
```ts
clientId: process.env.FACEBOOK_CLIENT_ID || "facebook_placeholder",
clientSecret: process.env.FACEBOOK_CLIENT_SECRET || "facebook_placeholder",
```
เป็น:
```ts
clientId: process.env.FACEBOOK_CLIENT_ID || "",
clientSecret: process.env.FACEBOOK_CLIENT_SECRET || "",
```

เปลี่ยนบรรทัด 54-55 จาก:
```ts
clientId: process.env.LINE_CLIENT_ID || "line_placeholder",
clientSecret: process.env.LINE_CLIENT_SECRET || "line_placeholder",
```
เป็น:
```ts
clientId: process.env.LINE_CLIENT_ID || "",
clientSecret: process.env.LINE_CLIENT_SECRET || "",
```

- [ ] **Step 2: omise.ts — เปลี่ยน Omise placeholder**

เปลี่ยนบรรทัด 4-5 จาก:
```ts
publicKey: process.env.NEXT_PUBLIC_OMISE_PUBLIC_KEY || "pkey_test_placeholder",
secretKey: process.env.OMISE_SECRET_KEY || "skey_test_placeholder",
```
เป็น:
```ts
publicKey: process.env.NEXT_PUBLIC_OMISE_PUBLIC_KEY || "",
secretKey: process.env.OMISE_SECRET_KEY || "",
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add lib/auth.ts lib/omise.ts
git commit -m "fix: replace placeholder strings with empty fallback in auth and omise"
```

---

### Task 4: ลบ placeholder URL ใน ProfileSection + extract provider config

**Files:**
- Modify: `app/account/ProfileSection.tsx:59`
- Modify: `lib/config.ts` (เพิ่ม OAUTH_PROVIDERS)
- Modify: `app/api/account/providers/route.ts:12-16`

- [ ] **Step 1: ProfileSection — เปลี่ยน placeholder avatar URL**

เปลี่ยนบรรทัด 59 จาก:
```tsx
placeholder="https://example.com/avatar.jpg"
```
เป็น:
```tsx
placeholder=""
```

- [ ] **Step 2: lib/config.ts — เพิ่ม OAUTH_PROVIDERS constant**

เพิ่มหลังบรรทัด 10 (`DEFAULT_GENRES`) ใน `lib/config.ts`:
```ts
export const OAUTH_PROVIDERS = [
  { id: "google", name: "Google", icon: "google" },
  { id: "facebook", name: "Facebook", icon: "facebook" },
  { id: "line", name: "LINE", icon: "line" },
] as const;
```

- [ ] **Step 3: providers/route.ts — import จาก config**

เปลี่ยนบรรทัด 12-16 จาก:
```ts
const configuredProviders = [
  { id: "google", name: "Google", icon: "google" },
  { id: "facebook", name: "Facebook", icon: "facebook" },
  { id: "line", name: "LINE", icon: "line" },
];
```
เป็น:
```ts
import { OAUTH_PROVIDERS } from "@/lib/config";

// ในฟังก์ชัน GET:
const configuredProviders = OAUTH_PROVIDERS.map((p) => ({ ...p }));
```

- [ ] **Step 4: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add app/account/ProfileSection.tsx lib/config.ts app/api/account/providers/route.ts
git commit -m "fix: remove placeholder avatar URL, extract OAuth providers to config"
```

---

### Task 5: เปลี่ยน NEXTAUTH_SECRET ใน .env

**Files:**
- Modify: `.env:4`

- [ ] **Step 1: Generate new secret**

Run: `openssl rand -base64 32`
เก็บ output ไว้

- [ ] **Step 2: แก้ .env**

เปลี่ยนบรรทัด 4 จาก:
```
NEXTAUTH_SECRET="harmoniq-super-secret-key-12345"
```
เป็น (ใช้ค่าจาก step 1):
```
NEXTAUTH_SECRET="<generated-key>"
```

- [ ] **Step 3: ตรวจสอบ .gitignore ครอบ .env**

Run: `grep "\.env" .gitignore`
Expected: ✅ ได้รับการยืนยันแล้ว — `.env` อยู่ใน `.gitignore` แล้ว (`git ls-files .env` ไม่มี output)

- [ ] **Step 4: ไม่ commit (local-only change)**

`.env` ถูก gitignore — การแก้ครั้งนี้เป็น local-only ไม่ต้อง commit และไม่ต้องแก้ `.gitignore` (ไม่มี `.env.example` อยู่ — อย่าสร้างใหม่ตาม YAGNI)

---

### Task 6: เปลี่ยน seed password เป็น random

**Files:**
- Modify: `prisma/seed.ts:8`

- [ ] **Step 1: เพิ่ม crypto import**

เพิ่มที่บรรทัด 2:
```ts
import crypto from "crypto";
```

- [ ] **Step 2: เปลี่ยน password จาก hardcoded เป็น random**

เปลี่ยนบรรทัด 8 จาก:
```ts
const password = "adminpassword123";
```
เป็น:
```ts
const password = crypto.randomBytes(16).toString("hex");
```

- [ ] **Step 3: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add prisma/seed.ts
git commit -m "security: generate random admin password in seed script"
```

---

### Task 7: Clean up useStemJob.ts (unused file — minimal fix)

**Files:**
- Modify: `app/components/studio/useStemJob.ts:141-155`

- [ ] **Step 1: ลบ progress simulation**

เปลี่ยน `startProgressSimulation` (lines 141-147) และ `stopProgressSimulation` (lines 149-155) เป็น dummy:

```ts
const startProgressSimulation = () => {
  setProgress(0);
};

const stopProgressSimulation = (_finalProgress?: number) => {
  setProgress(0);
};
```

- [ ] **Step 2: Type-check**

Run: `npx tsc --noEmit`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add app/components/studio/useStemJob.ts
git commit -m "fix: remove fake progress simulation in useStemJob"
```

---

### Task 8: แก้ test assertion string ใน test_auth_guard.py

**Files:**
- Modify: `backend/tests/test_auth_guard.py:10,16,26`

- [ ] **Step 1: แก้ test_free_tier_cannot_use_cnn_model**

เปลี่ยนบรรทัด 10 จาก:
```python
self.assertIn("CNN model requires Basic or Pro", cm.exception.detail)
```
เป็น:
```python
self.assertIn("โมเดล AutoEQ แบบ CNN สงวนสิทธิ์เฉพาะผู้ใช้สมาชิกระดับ Basic หรือ Pro", cm.exception.detail)
```

- [ ] **Step 2: แก้ test_free_tier_exceeded_quota**

เปลี่ยนบรรทัด 16 จาก:
```python
self.assertIn("Monthly quota reached", cm.exception.detail)
```
เป็น:
```python
self.assertIn("โควตาประมวลผลฟรีสำหรับผู้ใช้ FREE เต็มแล้ว", cm.exception.detail)
```

- [ ] **Step 3: แก้ test_pitch_shift_range_limit_free_tier**

เปลี่ยนบรรทัด 26 จาก:
```python
self.assertIn("Pitch shift of 5 semitones exceeds allowed limit", cm.exception.detail)
```
เป็น:
```python
self.assertIn("การปรับ Pitch 5 เซมิโทน เกินโควตาของแพ็กเกจ FREE", cm.exception.detail)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest backend/tests/test_auth_guard.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_auth_guard.py
git commit -m "fix: correct test assertion strings to match Thai error messages"
```

---

### Self-Review

| Check | Result |
|-------|--------|
| Spec coverage | ✅ ทั้ง 8 requirements มี task รับผิดชอบ |
| No placeholders | ✅ ทุก step มีโค้ดจริง ไม่มี TBD/TODO |
| Type consistency | ✅ ไม่มีการพึ่งพาข้าม task — ทำ parallel ได้ |
| idle code | ✅ useStemJob.ts ไม่ถูก import แต่ทำ minimal cleanup แทนการ refactor ใหญ่ |
