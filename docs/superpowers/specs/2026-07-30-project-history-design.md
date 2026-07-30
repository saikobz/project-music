# Project History (ประวัติการแยกแทร็กเสียง) — Design Spec

## 1. Overview

ทำให้ระบบประวัติการแยกแทร็กเสียง (`/dashboard/history`) ทำงานได้จริง แทน mock data เดิม โดยบันทึกประวัติการประมวลผลทุกรูปแบบ (separate, EQ, compressor, pitch-shift, analyze) ลง SQLite ผ่าน Prisma และแสดงผลบนหน้า history พร้อมความสามารถดาวน์โหลดซ้ำสำหรับรายการที่เป็น stem separation

## 2. Data Model

เพิ่ม Model `ProjectRecord` ใน `prisma/schema.prisma`:

```prisma
model ProjectRecord {
  id               String   @id @default(cuid())
  userId           String
  user             User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  action           String   // "separate" | "apply-eq-ai" | "apply-compressor" | "pitch-shift" | "analyze"
  originalFilename String
  fileId           String?  // backend file_id (เฉพาะ action ที่มีไฟล์ output)
  stems            String?  // JSON string ["Vocals","Drums","Bass","Other"]
  createdAt        DateTime @default(now())
}
```

- `action` เก็บประเภท action จาก `useStemJob.ts`
- `fileId` ใช้สำหรับ reconstruct download URL เฉพาะ `separate` (และ future `apply-eq-ai`, `pitch-shift` ที่มี file_id)
- `stems` เก็บเป็น JSON string เฉพาะ action `separate`

## 3. API Routes (Next.js App Router)

### `POST /api/history`
- Protected (requires valid session)
- Body: `{ action: string, originalFilename: string, fileId?: string, stems?: string[] }`
- Creates `ProjectRecord` ใน DB
- Returns `{ status: "success", record: ProjectRecord }`

### `GET /api/history`
- Protected
- Query params: ไม่มี (ดึงทั้งหมด)
- Returns `{ records: ProjectRecord[] }` เรียงตาม `createdAt` desc

### `DELETE /api/history/[id]`
- Protected
- ตรวจสอบว่า record เป็นของ user ปัจจุบันก่อนลบ
- Returns `{ status: "success" }`

## 4. Frontend — บันทึกประวัติ (useStemJob.ts)

หลังแต่ละ action สำเร็จ (ในบล็อก `response.data.status === "success"`), fire-and-forget `POST /api/history`:

- **separate**: `{ action: "separate", originalFilename: file.name, fileId: data.file_id, stems: ["Vocals","Drums","Bass","Other"] }`
- **apply-eq-ai**: `{ action: "apply-eq-ai", originalFilename: file.name }`
- **apply-compressor**: `{ action: "apply-compressor", originalFilename: file.name }`
- **pitch-shift**: `{ action: "pitch-shift", originalFilename: file.name }`
- **analyze**: `{ action: "analyze", originalFilename: file.name }`

ใช้ `fetch` แบบไม่ `await` / ไม่ `toast.error` ถ้าล้มเหลว — ไม่รบกวน UX หลัก

## 5. Frontend — หน้า History (`app/dashboard/history/page.tsx`)

### Data Flow
- แทน mock `useState<HistoryItem[]>`, เปลี่ยนเป็น `fetch("/api/history")` ใน `useEffect`
- State เดิม (`data`, `loading`) สำหรับ `/api/account` ยังคงอยู่เพื่อ auth guard
- เพิ่ม state `records` และ `recordsLoading`

### การแสดงผล

| Column | รายละเอียด |
|--------|------------|
| ชื่อไฟล์ + Play | ทุก action แสดงชื่อไฟล์ + duration (ถ้ามี) ปุ่ม play ใช้งานได้เฉพาะ **separate** (ต่อกับ `/separated/{fileId}/vocals.wav` หรือ main stem) |
| วันที่ | `createdAt` format เป็นวันที่ไทย |
| ประเภท / รายละเอียด | **separate** → แสดง stems badges, **อื่น ๆ** → แสดง action type badge (สีต่างกัน) |
| การจัดการ | **separate** → download จริง (`/download/{fileId}`) + delete, **อื่น ๆ** → delete เท่านั้น |

### Empty State + Auth Guard
- ไม่เปลี่ยนแปลง: คงเดิมทั้ง `loading` (skeleton), `!data` (sign in prompt), `records.length === 0` (empty state)

### Play (เฉพาะ separate)
- ใช้ `<audio>` element หรือ WaveSurfer.js เล่น stem หลักจาก `/separated/{fileId}/vocals.wav`
- Mock `playingId` toggle เดิม → เปลี่ยนเป็น audio play/pause จริง

### Download (เฉพาะ separate)
- แทน `alert()` → `window.open(API_BASE + "/download/" + fileId, "_blank")` หรือสร้าง `<a>` download

### Delete
- แทน local state filter → `DELETE /api/history/[id]` → ลบจาก local state หลัง response success

## 6. ไม่มีการเปลี่ยนแปลง Backend

Backend (FastAPI) ไม่มีการเปลี่ยนแปลงใด ๆ เพราะ:
- `/download/{file_id}` และ `/separated/{file_id}/{filename}` มีอยู่แล้ว
- `POST /api/history` ทำงานบน Next.js API route ซึ่งมี Prisma access
- การบันทึกทำจาก frontend หลังได้รับ response success

## 7. Testing

### Frontend (Jest)
- ทดสอบ API route `/api/history`: GET, POST, DELETE พร้อม session guard
- ทดสอบ history page render ด้วย mock records
- ทดสอบ empty state, auth state

### ไม่ต้องเพิ่ม backend test (ไม่มี backend change)

## 8. Files ที่ต้องแก้ไข/สร้าง

| File | Action |
|------|--------|
| `prisma/schema.prisma` | เพิ่ม `ProjectRecord` model |
| `app/api/history/route.ts` | **สร้าง** — POST + GET |
| `app/api/history/[id]/route.ts` | **สร้าง** — DELETE |
| `app/components/studio/useStemJob.ts` | เพิ่ม `fetch("/api/history", { method: "POST" })` หลัง success แต่ละ action |
| `app/dashboard/history/page.tsx` | เปลี่ยน mock → real data, download จริง, play จริง, delete จริง |

## 9. ขอบเขต (Out of Scope)
- ไม่มีการเพิ่ม download/play สำหรับ non-separate actions
- ไม่มีการเพิ่ม real-time update / websocket
- ไม่มีการ cleanup backend files เมื่อลบ history record (TTL cleanup ที่ backend จัดการอยู่แล้ว)
- ไม่มีการแก้ไข backend
