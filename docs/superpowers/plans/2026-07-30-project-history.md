# Project History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mock data on `/dashboard/history` with real database-backed records + wire up download/play/delete

**Architecture:** Add `ProjectRecord` Prisma model → Next.js API routes (`POST/GET/DELETE /api/history`) → frontend saves from `useStemJob.ts` after success → history page fetches real data

**Tech Stack:** Prisma (SQLite), Next.js App Router API, React 19, TypeScript

## Global Constraints

- Python 3.10 compatibility (no backend changes)
- All DB operations via Next.js API routes (backend is stateless)
- Fire-and-forget save — no toast/UX interruption on save failure
- Follow existing pattern from `app/api/account/route.ts` for API routes
- Tests in `tests/` dir using ts-jest

---

### Task 1: Prisma Model + Migration

**Files:**
- Modify: `prisma/schema.prisma`

- [ ] **Step 1: Add ProjectRecord model to schema.prisma**

Insert before the closing of the file (after `UsageQuota` model):

```prisma
model ProjectRecord {
  id               String   @id @default(cuid())
  userId           String
  user             User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  action           String
  originalFilename String
  fileId           String?
  stems            String?
  createdAt        DateTime @default(now())
}
```

- [ ] **Step 2: Run Prisma migration**

```bash
npx prisma migrate dev --name add_project_record
npx prisma generate
```

Expected output: Migration `xxxxxx_add_project_record` applied, Prisma client regenerated.

- [ ] **Step 3: Commit**

```bash
git add prisma/schema.prisma prisma/migrations/
git commit -m "feat: add ProjectRecord model for project history"
```

---

### Task 2: API Routes (POST + GET + DELETE)

**Files:**
- Create: `app/api/history/route.ts`
- Create: `app/api/history/[id]/route.ts`

**Interfaces:**
- Consumes: `prisma` from `@/lib/prisma`, `authOptions` from `@/lib/auth`, `getServerSession` from `next-auth`
- Produces: `GET /api/history` → `{ records: ProjectRecord[] }`, `POST /api/history` → `{ status: "success", record: ProjectRecord }`, `DELETE /api/history/[id]` → `{ status: "success" }`

- [ ] **Step 1: Create `app/api/history/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const records = await prisma.projectRecord.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
  });

  return NextResponse.json({ records });
}

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { action, originalFilename, fileId, stems } = body;

  if (!action || !originalFilename) {
    return NextResponse.json(
      { error: "Missing required fields: action, originalFilename" },
      { status: 400 }
    );
  }

  const record = await prisma.projectRecord.create({
    data: {
      userId: session.user.id,
      action,
      originalFilename,
      fileId: fileId || null,
      stems: stems ? JSON.stringify(stems) : null,
    },
  });

  return NextResponse.json({ status: "success", record });
}
```

- [ ] **Step 2: Create `app/api/history/[id]/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;

  const record = await prisma.projectRecord.findUnique({ where: { id } });
  if (!record) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  if (record.userId !== session.user.id) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  await prisma.projectRecord.delete({ where: { id } });
  return NextResponse.json({ status: "success" });
}
```

- [ ] **Step 3: Commit**

```bash
git add app/api/history/
git commit -m "feat: add Project History API routes (GET, POST, DELETE)"
```

---

### Task 3: API Route Tests

**Files:**
- Create: `tests/history.test.ts`

- [ ] **Step 1: Create test file**

```typescript
jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

import { GET, POST } from "../app/api/history/route";
import { getServerSession } from "next-auth";

describe("History API — GET", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GET();
    expect(res.status).toBe(401);
  });
});

describe("History API — POST", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({ action: "separate", originalFilename: "test.wav" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if missing required fields", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "test@test.com" },
    });
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
  });
});
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
npx jest tests/history.test.ts --verbose
```

Expected: All 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/history.test.ts
git commit -m "test: add Project History API route tests"
```

---

### Task 4: Save History from useStemJob.ts

**Files:**
- Modify: `app/components/studio/useStemJob.ts`

- [ ] **Step 1: Add saveHistory helper + call after each success**

In `useStemJob.ts`, add a helper function after the `validateAndSetFile` function (around line 106):

```typescript
const saveHistory = (action: string, fileId?: string, stems?: string[]) => {
  fetch("/api/history", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      originalFilename: file?.name || "unknown.wav",
      ...(fileId && { fileId }),
      ...(stems && { stems }),
    }),
  }).catch(() => {
    /* silent — don't interrupt user flow */
  });
};
```

Then, after each success block in `handleSubmit`:

**After separate success** (around line 266, after `toast.success("แยกเสียงสำเร็จ!")`):
```typescript
saveHistory("separate", data.file_id, ["Vocals", "Drums", "Bass", "Other"]);
```

**After apply-eq-ai success** (around line 289, after `toast.success("ประมวลผลเสียงสำเร็จ!")`):
```typescript
saveHistory("apply-eq-ai");
```

**After apply-compressor success** (same area):
```typescript
saveHistory("apply-compressor");
```

**After pitch-shift success** (same area):
```typescript
saveHistory("pitch-shift");
```

**After analyze success** (around line 258, after `toast.success("วิเคราะห์ไฟล์เสียงสำเร็จ!")`):
```typescript
saveHistory("analyze");
```

Note: The non-separate actions (`apply-eq-ai`, `apply-compressor`, `pitch-shift`) all share the same success flow (around line 269-289 in the current file). Add `saveHistory` call there. The `analyze` success is at line 257-258. The `separate` success is at line 259-268.

- [ ] **Step 2: Verify TypeScript compiles**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add app/components/studio/useStemJob.ts
git commit -m "feat: save project history after successful audio processing"
```

---

### Task 5: History Page — Real Data + Real Actions

**Files:**
- Modify: `app/dashboard/history/page.tsx`

- [ ] **Step 1: Replace mock data with real fetch + add new imports**

Replace the imports to add:

```typescript
import { API_BASE_URL } from "@/lib/config";
// Keep all existing imports from lucide-react etc.
```

Replace the `interface HistoryItem` with:

```typescript
interface HistoryRecord {
  id: string;
  action: string;
  originalFilename: string;
  fileId: string | null;
  stems: string | null;
  createdAt: string;
}
```

Replace the mock `useState<HistoryItem[]>` and the second `useEffect` that fetches `/api/account`:

Keep the `/api/account` fetch for auth guard. Add a second fetch for records:

```typescript
const [records, setRecords] = useState<HistoryRecord[]>([]);
const [recordsLoading, setRecordsLoading] = useState(true);

useEffect(() => {
  fetch("/api/account")
    .then((res) => res.json())
    .then(setData)
    .catch(() => setData(null))
    .finally(() => setLoading(false));
}, []);

useEffect(() => {
  if (!data || data.error) {
    setRecordsLoading(false);
    return;
  }
  fetch("/api/history")
    .then((res) => res.json())
    .then((d) => setRecords(d.records || []))
    .catch(() => setRecords([]))
    .finally(() => setRecordsLoading(false));
}, [data]);
```

- [ ] **Step 2: Replace delete handler**

```typescript
const handleDelete = async (id: string) => {
  const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
  if (res.ok) {
    setRecords((prev) => prev.filter((r) => r.id !== id));
  }
};
```

- [ ] **Step 3: Replace download handler**

```typescript
const handleDownload = (fileId: string) => {
  const url = `${API_BASE_URL}/download/${fileId}`;
  const a = document.createElement("a");
  a.href = url;
  a.download = "separated.zip";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};
```

- [ ] **Step 4: Replace play button with real audio playback**

Replace the togglePlay logic and the play button:

```typescript
const [playingId, setPlayingId] = useState<string | null>(null);
const audioRef = useRef<HTMLAudioElement | null>(null);

const togglePlay = (fileId: string) => {
  if (playingId === fileId) {
    audioRef.current?.pause();
    setPlayingId(null);
  } else {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    const url = `${API_BASE_URL}/separated/${fileId}/vocals.wav`;
    const audio = new Audio(url);
    audio.onended = () => setPlayingId(null);
    audio.play().catch(() => {
      /* file may be cleaned up — silent fail */
      setPlayingId(null);
    });
    audioRef.current = audio;
    setPlayingId(fileId);
  }
};
```

- [ ] **Step 5: Replace the table rendering to use real records**

Key changes in the table body (existing line ~150):

```tsx
{records.map((record) => {
  const stemsList: string[] = record.stems ? JSON.parse(record.stems) : [];
  const isSeparate = record.action === "separate";
  const actionLabels: Record<string, string> = {
    separate: "Stem Separation",
    "apply-eq-ai": "Auto-EQ",
    "apply-compressor": "Compressor",
    "pitch-shift": "Pitch Shift",
    analyze: "Audio Analysis",
  };

  return (
    <tr key={record.id} className="hover:bg-[#161616] transition-colors">
      <td className="py-4 px-4">
        <div className="flex items-center gap-3">
          {isSeparate && (
            <button
              onClick={() => togglePlay(record.fileId!)}
              className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center hover:bg-purple-500 hover:text-white transition"
            >
              {playingId === record.fileId ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
          )}
          <div>
            <p className="font-semibold text-white truncate max-w-xs">{record.originalFilename}</p>
            <span className="text-[11px] text-purple-400">{actionLabels[record.action] || record.action}</span>
          </div>
        </div>
      </td>
      <td className="py-4 px-4 text-[#A0A0A0] text-xs">
        {new Date(record.createdAt).toLocaleDateString("th-TH", {
          year: "numeric",
          month: "short",
          day: "numeric",
        })}
      </td>
      <td className="py-4 px-4">
        {isSeparate && stemsList.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {stemsList.map((stem) => (
              <span
                key={stem}
                className="px-2 py-0.5 rounded-md bg-[#202020] border border-[#303030] text-[11px] text-purple-300"
              >
                {stem}
              </span>
            ))}
          </div>
        ) : (
          <span className="text-[11px] text-[#555]">—</span>
        )}
      </td>
      <td className="py-4 px-4 text-right">
        <div className="flex items-center justify-end gap-2">
          {isSeparate && record.fileId && (
            <button
              onClick={() => handleDownload(record.fileId!)}
              className="p-2 rounded-lg bg-[#202020] hover:bg-[#303030] text-white transition text-xs flex items-center gap-1.5"
            >
              <Download className="w-4 h-4 text-purple-400" />
              <span className="hidden sm:inline">ดาวน์โหลด</span>
            </button>
          )}
          <button
            onClick={() => handleDelete(record.id)}
            className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </td>
    </tr>
  );
})}
```

- [ ] **Step 6: Update the empty state condition**

Change all references from `historyItems.length === 0` to `records.length === 0`.

Also update the table `thead` to change "สเต็มที่มี" to "รายละเอียด" to better reflect mixed content.

- [ ] **Step 7: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 8: Run frontend tests**

```bash
npx jest --verbose
```

Expected: All existing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add app/dashboard/history/page.tsx
git commit -m "feat: wire history page to real API data with download/play/delete"
```

---

## Verification Checklist

Before marking complete:
- [ ] All 5 tasks committed
- [ ] `npx tsc --noEmit` passes
- [ ] `npx jest` passes
- [ ] `npx prisma studio` shows `ProjectRecord` table with test data after running a separation
- [ ] `/dashboard/history` shows real records (not mock)
- [ ] Delete removes record from DB and UI
- [ ] Download works for `separate` records
- [ ] Play works for `separate` records (vocals stem)
