# Account Data Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add export user data and delete account functionality to the account settings page, with proper password verification and Omise cleanup.

**Architecture:** Add `DELETE` handler to `app/api/account/route.ts` for account deletion → create `GET /api/account/export` for data download → add `DataSection` component with export button and delete account modal → register as 5th tab in account page.

**Tech Stack:** Prisma (SQLite), Next.js App Router API, React 19, TypeScript, Tailwind CSS 4, lucide-react, sonner, bcryptjs

## Global Constraints

- All DB operations via Next.js API routes (backend is stateless)
- Follow existing patterns from `app/api/account/route.ts` for API routes
- Use `sonner` toast for notifications
- Use `lucide-react` icons
- Account deletion requires password verification (if user has password) — OAuth users just confirm
- Cascade delete handles all related models (Account, Session, Subscription, UsageQuota, ProjectRecord)
- Omise schedule/customer cleanup is best-effort — don't block deletion on Omise failures

---

### Task 3.1: API — DELETE /api/account (delete account)

**Files:**
- Modify: `app/api/account/route.ts` (add `import bcrypt from "bcryptjs"` + `import { omise } from "@/lib/omise"` + `DELETE` handler)

**Produces:** `DELETE /api/account` → `{ success: true }` after password verification and cascade delete

- [ ] **Step 1: Read current `app/api/account/route.ts`**

The file currently has GET handler. Add these imports at the top:
```typescript
import bcrypt from "bcryptjs";
import { omise } from "@/lib/omise";
```

- [ ] **Step 2: Add DELETE handler to the same file**

Append after the GET function:
```typescript
export async function DELETE(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { password } = await req.json();

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: { subscription: true },
  });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  if (user.password) {
    if (!password) {
      return NextResponse.json({ error: "Password is required to delete account" }, { status: 400 });
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return NextResponse.json({ error: "Incorrect password" }, { status: 401 });
    }
  }

  // Best-effort cleanup of Omise data
  if (user.subscription?.omiseScheduleId) {
    try {
      await omise.schedules.destroy(user.subscription.omiseScheduleId);
    } catch {
      // Schedule may already be destroyed — continue
    }
  }

  await prisma.user.delete({ where: { id: session.user.id } });

  return NextResponse.json({ success: true });
}
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/api/account/route.ts
git commit -m "feat: add DELETE account endpoint with password verification and Omise cleanup"
```

---

### Task 3.2: API — GET /api/account/export (export data)

**Files:**
- Create: `app/api/account/export/route.ts`

**Produces:** `GET /api/account/export` → JSON file download with all user data

- [ ] **Step 1: Create `app/api/account/export/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      subscription: true,
      usageQuotas: { orderBy: { periodStart: "desc" } },
      projectRecords: { orderBy: { createdAt: "desc" } },
      accounts: { select: { provider: true, providerAccountId: true, createdAt: true } },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const exportData = {
    exportedAt: new Date().toISOString(),
    profile: {
      name: user.name,
      email: user.email,
      createdAt: user.createdAt,
    },
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
    subscription: user.subscription
      ? {
          tier: user.subscription.tier,
          status: user.subscription.status,
          paymentMethod: user.subscription.paymentMethod,
          currentPeriodStart: user.subscription.currentPeriodStart,
          currentPeriodEnd: user.subscription.currentPeriodEnd,
        }
      : null,
    usageQuotas: user.usageQuotas.map((q) => ({
      monthlyQuota: q.monthlyQuota,
      usedCount: q.usedCount,
      periodStart: q.periodStart,
      periodEnd: q.periodEnd,
    })),
    projectHistory: user.projectRecords.map((r) => ({
      action: r.action,
      originalFilename: r.originalFilename,
      fileId: r.fileId,
      stems: r.stems,
      createdAt: r.createdAt,
    })),
    connectedAccounts: user.accounts.map((a) => ({
      provider: a.provider,
      linkedAt: a.createdAt,
    })),
  };

  const dateStr = new Date().toISOString().split("T")[0];
  return new NextResponse(JSON.stringify(exportData, null, 2), {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="harmoniq-export-${dateStr}.json"`,
    },
  });
}
```

- [ ] **Step 2: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add app/api/account/export/route.ts
git commit -m "feat: add user data export API endpoint"
```

---

### Task 3.3: Frontend — Data Section (export + delete account)

**Files:**
- Create: `app/account/DataSection.tsx`
- Modify: `app/account/page.tsx` (add import + tab + reference)

- [ ] **Step 1: Create `app/account/DataSection.tsx`**

```typescript
"use client";
import React, { useState } from "react";
import { signOut } from "next-auth/react";
import { toast } from "sonner";
import { Download, AlertTriangle, Trash2 } from "lucide-react";

interface DataSectionProps {
  hasPassword: boolean;
}

export default function DataSection({ hasPassword }: DataSectionProps) {
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleting, setDeleting] = useState(false);

  const handleExport = () => {
    const a = document.createElement("a");
    a.href = "/api/account/export";
    a.download = `harmoniq-export-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast.success("Exporting your data...");
  };

  const handleDelete = async () => {
    if (!window.confirm("This will permanently delete your account and all data. Are you sure?")) return;
    setDeleting(true);
    try {
      const res = await fetch("/api/account", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: deletePassword || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to delete account");
        return;
      }
      toast.success("Account deleted");
      signOut({ callbackUrl: "/" });
    } catch {
      toast.error("Network error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Export Section */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Export Data</p>
        <div className="p-4 rounded-xl bg-[#1A1A1A] border border-[#222] flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">Download Your Data</p>
            <p className="text-xs text-[#666]">Export profile, preferences, history, and usage data as JSON.</p>
          </div>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-[#34D399]/10 text-[#34D399] text-xs font-semibold rounded-lg hover:bg-[#34D399]/20 transition cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Delete Account Section */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Danger Zone</p>
        <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
          <div className="flex items-start gap-3 mb-4">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-red-400">Delete Account</p>
              <p className="text-xs text-[#888] mt-1">
                Permanently delete your account, all data, and project history. This action cannot be undone.
              </p>
            </div>
          </div>
          {!showDeleteModal ? (
            <button
              onClick={() => setShowDeleteModal(true)}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-400 text-xs font-semibold rounded-lg hover:bg-red-500/20 transition cursor-pointer"
            >
              <Trash2 className="w-4 h-4" />
              Delete My Account
            </button>
          ) : (
            <div className="space-y-3 border-t border-red-500/10 pt-4">
              {hasPassword && (
                <div>
                  <label className="block text-xs text-[#888] mb-1">Enter your password to confirm:</label>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(e) => setDeletePassword(e.target.value)}
                    className="w-full max-w-xs bg-[#0A0A0A] border border-red-500/30 rounded-lg px-3 py-2 text-sm text-[#F3F3F3] focus:outline-none focus:border-red-400 transition"
                  />
                </div>
              )}
              <div className="flex items-center gap-3">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg transition cursor-pointer"
                >
                  {deleting ? "Deleting..." : "Confirm Delete"}
                </button>
                <button
                  onClick={() => { setShowDeleteModal(false); setDeletePassword(""); }}
                  className="px-4 py-2 bg-[#1A1A1A] text-[#888] text-xs rounded-lg hover:text-white transition cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Update `app/account/page.tsx`**

Add to imports:
```typescript
import DataSection from "./DataSection";
```

Add the "data" tab to the TABS array:
```typescript
{ id: "data", label: "Data", icon: <Trash2 className="w-4 h-4" /> },
```

Update the `AccountTab` type:
```typescript
type AccountTab = "profile" | "password" | "accounts" | "preferences" | "data";
```

Add in tab content section:
```typescript
{activeTab === "data" && <DataSection hasPassword={data.user.hasPassword} />}
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/account/DataSection.tsx app/account/page.tsx
git commit -m "feat: add data section with export and delete account"
```

---

### Task 3.4: Tests

**Files:**
- Modify: `tests/account.test.ts`

Add the omise mock and bcrypt mock (may already exist):

```typescript
jest.mock("@/lib/omise", () => ({
  omise: {
    schedules: { destroy: jest.fn() },
  },
}));
```

Add test blocks:

```typescript
// ... inside describe blocks ...
describe("Account API — DELETE /api/account", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account", {
      method: "DELETE",
      body: JSON.stringify({ password: "test123" }),
    });
    const res = await DELETE(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if password missing for credential user", async () => {
    const { prisma: p } = require("@/lib/prisma");
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    p.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: "hashed", subscription: null });
    const req = new Request("http://localhost/api/account", {
      method: "DELETE",
      body: JSON.stringify({}),
    });
    const res = await DELETE(req);
    expect(res.status).toBe(400);
  });
});

describe("Account API — GET /api/account/export", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GetExport();
    expect(res.status).toBe(401);
  });
});
```

Also add the import for DELETE and GetExport:
```typescript
import { DELETE } from "../app/api/account/route";
import { GET as GetExport } from "../app/api/account/export/route";
```

Run: `npx jest tests/account.test.ts --verbose`

Commit: `test: add delete account and export API tests`

---

## Verification Checklist

- [ ] `npx tsc --noEmit` passes
- [ ] `npx jest` passes
- [ ] Export downloads a JSON file with all user data
- [ ] Delete account requires password (for credential users), cascades all data
- [ ] Omise schedule is destroyed on delete (if exists)
- [ ] After deletion, user is signed out and redirected to home
