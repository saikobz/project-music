# Account Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `/account` from a read-only dashboard into a full settings page with profile editing, password change, connected accounts view, and user preferences (theme, language, notifications).

**Architecture:** Add preference fields to Prisma `User` model → add `Account`/`Session`/`VerificationToken` models for PrismaAdapter → create `PUT` API routes for profile, password, preferences → refactor `/account` page into a tabbed layout with section components.

**Tech Stack:** Prisma (SQLite), Next.js App Router API, React 19, TypeScript, Tailwind CSS 4, lucide-react, sonner, bcryptjs

## Global Constraints

- Python 3.10 compatibility (no backend changes)
- All DB operations via Next.js API routes (backend is stateless)
- Follow existing patterns from `app/api/account/route.ts` for API routes
- Use `sonner` toast for success/error notifications (already in project)
- Use `lucide-react` icons (already in project)
- Theme change applies immediately via DOM mutation but defaults to `dark` on SSR
- Tests in `tests/` dir using ts-jest

---

### Task 1: Prisma — Add preference fields + Account/Session/VerificationToken models

**Files:**
- Modify: `prisma/schema.prisma`
- Create: `prisma/migrations/xxxxxx_add_preferences_and_account_models/`

- [ ] **Step 1: Add fields to User model + new models in schema.prisma**

Add to `model User` (after `omiseCustomerId` line):
```prisma
  theme             String   @default("DARK")
  language          String   @default("TH")
  emailNotifications Boolean @default(true)
```

Add after `model User` closing brace:
```prisma
model Account {
  id                String   @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String?
  access_token      String?
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String?
  session_state     String?
  user              User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
}

model VerificationToken {
  identifier String
  token      String   @unique
  expires    DateTime

  @@unique([identifier, token])
}
```

- [ ] **Step 2: Run Prisma migration**

```bash
npx prisma migrate dev --name add_preferences_and_account_models
npx prisma generate
```

Expected: Migration applied, Prisma client regenerated.

- [ ] **Step 3: Commit**

```bash
git add prisma/schema.prisma prisma/migrations/
git commit -m "feat: add preference fields and Account/Session models to Prisma schema"
```

---

### Task 2: Extend GET /api/account to return preferences + Create PUT /api/account/profile

**Files:**
- Modify: `app/api/account/route.ts`
- Create: `app/api/account/profile/route.ts`

**Interfaces:**
- Consumes: `authOptions` from `@/lib/auth`, `prisma` from `@/lib/prisma`, `getServerSession` from `next-auth`
- Produces: `PUT /api/account/profile` → `{ user: { name, email, image } }`
- Modified `GET /api/account` now also returns `preferences: { theme, language, emailNotifications }`

- [ ] **Step 1: Modify GET /api/account to include preference fields**

In `app/api/account/route.ts`, change the return object to include preferences:

```typescript
export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      subscription: true,
      usageQuotas: {
        orderBy: { periodStart: "desc" },
        take: 1,
      },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const currentQuota = user.usageQuotas[0] || {
    monthlyQuota: user.subscription?.tier === "PRO" ? -1 : user.subscription?.tier === "BASIC" ? 15 : 3,
    usedCount: 0,
  };

  return NextResponse.json({
    user: {
      id: user.id,
      name: user.name,
      email: user.email,
      image: user.image,
      createdAt: user.createdAt,
    },
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
    subscription: user.subscription || { tier: "FREE", status: "ACTIVE" },
    quota: currentQuota,
  });
}
```

- [ ] **Step 2: Create `app/api/account/profile/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function PUT(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { name, email, image } = await req.json();

  if (email && email !== session.user.email) {
    const existing = await prisma.user.findUnique({ where: { email } });
    if (existing) {
      return NextResponse.json({ error: "Email already in use" }, { status: 409 });
    }
  }

  const user = await prisma.user.update({
    where: { id: session.user.id },
    data: {
      ...(name !== undefined && { name }),
      ...(email !== undefined && { email }),
      ...(image !== undefined && { image }),
    },
  });

  return NextResponse.json({
    user: { name: user.name, email: user.email, image: user.image },
  });
}
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/api/account/route.ts app/api/account/profile/route.ts
git commit -m "feat: extend account API with profile update endpoint"
```

---

### Task 3: API — PUT /api/account/password

**Files:**
- Create: `app/api/account/password/route.ts`

**Interfaces:**
- Consumes: `bcrypt` from `bcryptjs`, `prisma`, `authOptions`, `getServerSession`
- Produces: `PUT /api/account/password` → `{ success: true }` or 400/401 error

- [ ] **Step 1: Create `app/api/account/password/route.ts`**

```typescript
import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function PUT(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { currentPassword, newPassword } = await req.json();

  if (!currentPassword || !newPassword) {
    return NextResponse.json(
      { error: "Current password and new password are required" },
      { status: 400 }
    );
  }

  if (newPassword.length < 6) {
    return NextResponse.json(
      { error: "New password must be at least 6 characters" },
      { status: 400 }
    );
  }

  const user = await prisma.user.findUnique({ where: { id: session.user.id } });
  if (!user || !user.password) {
    return NextResponse.json(
      { error: "Password change not available for OAuth-only accounts. Set a password first." },
      { status: 400 }
    );
  }

  const isValid = await bcrypt.compare(currentPassword, user.password);
  if (!isValid) {
    return NextResponse.json({ error: "Current password is incorrect" }, { status: 401 });
  }

  const hashedPassword = await bcrypt.hash(newPassword, 10);
  await prisma.user.update({
    where: { id: session.user.id },
    data: { password: hashedPassword },
  });

  return NextResponse.json({ success: true });
}
```

- [ ] **Step 2: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add app/api/account/password/route.ts
git commit -m "feat: add password change API endpoint"
```

---

### Task 4: API — GET /api/account/providers + PUT /api/account/preferences

**Files:**
- Create: `app/api/account/providers/route.ts`
- Create: `app/api/account/preferences/route.ts`

**Interfaces:**
- Produces: `GET /api/account/providers` → `{ providers: [{ id, name, linked, isPassword }] }`
- Produces: `PUT /api/account/preferences` → `{ preferences: { theme, language, emailNotifications } }`

- [ ] **Step 1: Create `app/api/account/providers/route.ts`**

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

  const configuredProviders = [
    { id: "google", name: "Google", icon: "google" },
    { id: "facebook", name: "Facebook", icon: "facebook" },
    { id: "line", name: "LINE", icon: "line" },
  ];

  const accounts = await prisma.account.findMany({
    where: { userId: session.user.id },
  });
  const linkedProviders = new Set(accounts.map((a) => a.provider));

  const user = await prisma.user.findUnique({ where: { id: session.user.id } });
  const hasPassword = !!user?.password;

  const providers = [
    ...configuredProviders.map((p) => ({
      ...p,
      linked: linkedProviders.has(p.id),
    })),
    {
      id: "credentials",
      name: "Email & Password",
      icon: "mail",
      linked: hasPassword,
    },
  ];

  return NextResponse.json({ providers });
}
```

- [ ] **Step 2: Create `app/api/account/preferences/route.ts`**

```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

const VALID_THEMES = ["DARK", "LIGHT"];
const VALID_LANGUAGES = ["TH", "EN"];

export async function PUT(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { theme, language, emailNotifications } = await req.json();

  if (theme !== undefined && !VALID_THEMES.includes(theme)) {
    return NextResponse.json({ error: "Invalid theme value" }, { status: 400 });
  }
  if (language !== undefined && !VALID_LANGUAGES.includes(language)) {
    return NextResponse.json({ error: "Invalid language value" }, { status: 400 });
  }

  const user = await prisma.user.update({
    where: { id: session.user.id },
    data: {
      ...(theme !== undefined && { theme }),
      ...(language !== undefined && { language }),
      ...(emailNotifications !== undefined && { emailNotifications }),
    },
  });

  return NextResponse.json({
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
  });
}
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/api/account/providers/route.ts app/api/account/preferences/route.ts
git commit -m "feat: add providers and preferences API endpoints"
```

---

### Task 5: Account page — Refactor to tabbed layout with Profile Section

**Files:**
- Modify: `app/account/page.tsx`
- Create: `app/account/tabs.ts` (tab IDs and labels)
- Create: `app/account/ProfileSection.tsx`

**Interfaces:**
- Consumes: `GET /api/account` data + `PUT /api/account/profile`
- Produces: Tabbed account page with Profile as default tab

- [ ] **Step 1: Create `app/account/tabs.ts`**

```typescript
export const ACCOUNT_TABS = [
  { id: "profile", label: "Profile", icon: "User" },
  { id: "password", label: "Password", icon: "KeyRound" },
  { id: "accounts", label: "Connected Accounts", icon: "Link" },
  { id: "preferences", label: "Preferences", icon: "Settings" },
] as const;

export type AccountTabId = (typeof ACCOUNT_TABS)[number]["id"];
```

- [ ] **Step 2: Create `app/account/ProfileSection.tsx`**

```typescript
"use client";
import React, { useState } from "react";
import { toast } from "sonner";
import { User, Mail, Camera } from "lucide-react";

interface ProfileSectionProps {
  user: { name: string | null; email: string; image: string | null };
  onUpdated: (updates: { name: string | null; email: string; image: string | null }) => void;
}

export default function ProfileSection({ user, onUpdated }: ProfileSectionProps) {
  const [name, setName] = useState(user.name || "");
  const [email, setEmail] = useState(user.email);
  const [image, setImage] = useState(user.image || "");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await fetch("/api/account/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name || null, email, image: image || null }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to update profile");
        return;
      }
      onUpdated(data.user);
      toast.success("Profile updated");
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Avatar */}
      <div className="flex items-center gap-4">
        {image ? (
          <img src={image} alt="Avatar" className="w-16 h-16 rounded-full object-cover border border-[#333]" />
        ) : (
          <div className="w-16 h-16 rounded-full bg-[#222] flex items-center justify-center">
            <User className="w-8 h-8 text-[#666]" />
          </div>
        )}
        <div className="flex-1">
          <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
            Avatar URL
          </label>
          <input
            type="url"
            value={image}
            onChange={(e) => setImage(e.target.value)}
            placeholder="https://example.com/avatar.jpg"
            className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg px-3 py-2 text-sm text-[#F3F3F3] placeholder-[#555] focus:outline-none focus:border-[#34D399] transition"
          />
        </div>
      </div>

      {/* Name */}
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Display Name
        </label>
        <div className="relative">
          <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#555]" />
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg pl-10 pr-3 py-2.5 text-sm text-[#F3F3F3] placeholder-[#555] focus:outline-none focus:border-[#34D399] transition"
          />
        </div>
      </div>

      {/* Email */}
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Email
        </label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#555]" />
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg pl-10 pr-3 py-2.5 text-sm text-[#F3F3F3] placeholder-[#555] focus:outline-none focus:border-[#34D399] transition"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={saving}
        className="px-6 py-2.5 bg-[#34D399] hover:bg-[#2cb984] disabled:opacity-50 text-[#0A0A0A] text-sm font-bold rounded-lg transition cursor-pointer"
      >
        {saving ? "Saving..." : "Save Changes"}
      </button>
    </form>
  );
}
```

- [ ] **Step 3: Rewrite `app/account/page.tsx` with tabbed layout**

```typescript
"use client";
import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { User, KeyRound, Link2, Settings } from "lucide-react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import ProfileSection from "./ProfileSection";

type AccountTab = "profile" | "password" | "accounts" | "preferences";

const TABS: { id: AccountTab; label: string; icon: React.ReactNode }[] = [
  { id: "profile", label: "Profile", icon: <User className="w-4 h-4" /> },
  { id: "password", label: "Password", icon: <KeyRound className="w-4 h-4" /> },
  { id: "accounts", label: "Connected Accounts", icon: <Link2 className="w-4 h-4" /> },
  { id: "preferences", label: "Preferences", icon: <Settings className="w-4 h-4" /> },
];

export default function AccountPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<AccountTab>("profile");

  const fetchAccount = useCallback(() => {
    setLoading(true);
    fetch("/api/account")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => { fetchAccount(); }, [fetchAccount]);

  const handleProfileUpdated = (updates: { name: string | null; email: string; image: string | null }) => {
    setData((prev: any) => ({
      ...prev,
      user: { ...prev.user, ...updates },
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center text-[#8E8E8E]">Loading account details...</div>
        <Footer />
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center space-y-4">
          <p className="text-red-400 font-medium">Please sign in to view your account details.</p>
          <Link href="/api/auth/signin" className="inline-block px-6 py-2 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] font-bold rounded-lg transition">
            Sign In
          </Link>
        </div>
        <Footer />
      </div>
    );
  }

  const { user, subscription, quota, preferences } = data;
  const used = quota.usedCount || 0;
  const max = quota.monthlyQuota;
  const isUnlimited = max === -1;
  const percent = isUnlimited ? 0 : Math.min(100, Math.round((used / max) * 100));

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow mx-auto w-full max-w-4xl px-4 py-12 space-y-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Account Settings</h1>
          <p className="text-[#8E8E8E] text-sm mt-1">Manage your profile, connected accounts, and preferences.</p>
        </div>

        {/* Tab Bar */}
        <div className="flex gap-1 border-b border-[#222] pb-0 overflow-x-auto">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-t-lg transition cursor-pointer whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-[#111] text-[#34D399] border border-b-0 border-[#222] -mb-[1px]"
                  : "text-[#888] hover:text-white hover:bg-[#111]"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Subscription & Quota Card (always visible) */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider">Current Subscription</p>
              <h3 className="text-xl font-extrabold text-[#34D399] mt-0.5">{subscription.tier} PLAN</h3>
            </div>
            <Link href="/pricing" className="px-4 py-2 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] text-xs font-bold rounded-lg transition">
              Change Plan
            </Link>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-[#CCCCCC]">Monthly Song Processing Quota</span>
            <span className="font-bold text-[#34D399]">{isUnlimited ? "Unlimited" : `${used} / ${max} songs used`}</span>
          </div>
          {!isUnlimited && (
            <div className="w-full bg-[#222222] rounded-full h-2.5 overflow-hidden">
              <div className="bg-[#34D399] h-2.5 rounded-full transition-all duration-300" style={{ width: `${percent}%` }}></div>
            </div>
          )}
        </div>

        {/* Tab Content */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6">
          {activeTab === "profile" && (
            <ProfileSection user={user} onUpdated={handleProfileUpdated} />
          )}
          {activeTab === "password" && (
            <p className="text-[#8E8E8E] text-sm">Password section coming in next task.</p>
          )}
          {activeTab === "accounts" && (
            <p className="text-[#8E8E8E] text-sm">Connected accounts section coming in next task.</p>
          )}
          {activeTab === "preferences" && (
            <p className="text-[#8E8E8E] text-sm">Preferences section coming in next task.</p>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
```

- [ ] **Step 4: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add app/account/page.tsx app/account/tabs.ts app/account/ProfileSection.tsx
git commit -m "feat: add tabbed account settings layout with profile edit section"
```

---

### Task 6: Account page — Password Change Section

**Files:**
- Create: `app/account/PasswordSection.tsx`
- Modify: `app/account/page.tsx` (replace placeholder with real component)

- [ ] **Step 1: Create `app/account/PasswordSection.tsx`**

```typescript
"use client";
import React, { useState } from "react";
import { toast } from "sonner";
import { KeyRound, Eye, EyeOff } from "lucide-react";

interface PasswordSectionProps {
  hasPassword: boolean;
}

export default function PasswordSection({ hasPassword }: PasswordSectionProps) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [show, setShow] = useState(false);
  const [saving, setSaving] = useState(false);

  if (!hasPassword) {
    return (
      <div className="text-center py-8 space-y-3">
        <KeyRound className="w-10 h-10 text-[#444] mx-auto" />
        <p className="text-[#8E8E8E] text-sm">You signed up with OAuth and don&apos;t have a password yet.</p>
        <p className="text-[#666] text-xs">Set a password to enable password login.</p>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }
    if (newPassword.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }
    setSaving(true);
    try {
      const res = await fetch("/api/account/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword, newPassword }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to change password");
        return;
      }
      toast.success("Password changed successfully");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-md space-y-5">
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Current Password
        </label>
        <div className="relative">
          <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#555]" />
          <input
            type={show ? "text" : "password"}
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            required
            className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg pl-10 pr-10 py-2.5 text-sm text-[#F3F3F3] focus:outline-none focus:border-[#34D399] transition"
          />
          <button type="button" onClick={() => setShow(!show)} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#555] hover:text-white cursor-pointer">
            {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          New Password
        </label>
        <input
          type={show ? "text" : "password"}
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
          minLength={6}
          className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg px-3 py-2.5 text-sm text-[#F3F3F3] focus:outline-none focus:border-[#34D399] transition"
        />
      </div>

      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Confirm New Password
        </label>
        <input
          type={show ? "text" : "password"}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          minLength={6}
          className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg px-3 py-2.5 text-sm text-[#F3F3F3] focus:outline-none focus:border-[#34D399] transition"
        />
      </div>

      <button
        type="submit"
        disabled={saving}
        className="px-6 py-2.5 bg-[#34D399] hover:bg-[#2cb984] disabled:opacity-50 text-[#0A0A0A] text-sm font-bold rounded-lg transition cursor-pointer"
      >
        {saving ? "Changing..." : "Change Password"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Update `app/account/page.tsx` — replace password placeholder**

Replace the line:
```typescript
          {activeTab === "password" && (
            <p className="text-[#8E8E8E] text-sm">Password section coming in next task.</p>
          )}
```

With:
```typescript
          {activeTab === "password" && (
            <PasswordSection hasPassword={!!data.user.password} />
          )}
```

Also add the import at the top of `page.tsx`:
```typescript
import PasswordSection from "./PasswordSection";
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/account/PasswordSection.tsx app/account/page.tsx
git commit -m "feat: add password change section to account settings"
```

---

### Task 7: Account page — Connected Accounts Section

**Files:**
- Create: `app/account/ConnectedAccountsSection.tsx`
- Modify: `app/account/page.tsx` (replace placeholder)

- [ ] **Step 1: Create `app/account/ConnectedAccountsSection.tsx`**

```typescript
"use client";
import React, { useEffect, useState } from "react";
import { signIn } from "next-auth/react";
import { toast } from "sonner";
import { Link2, Unlink, Mail, Globe, Facebook } from "lucide-react";

interface Provider {
  id: string;
  name: string;
  icon: string;
  linked: boolean;
}

const PROVIDER_ICONS: Record<string, React.ReactNode> = {
  google: <Globe className="w-5 h-5" />,
  facebook: <Facebook className="w-5 h-5" />,
  line: (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm3.5 13.5c-.28.28-.72.28-1 0l-2.5-2.5-2.5 2.5c-.28.28-.72.28-1 0s-.28-.72 0-1l3-3c.28-.28.72-.28 1 0l3 3c.28.28.28.72 0 1z"/>
    </svg>
  ),
  mail: <Mail className="w-5 h-5" />,
};

export default function ConnectedAccountsSection() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/account/providers")
      .then((res) => res.json())
      .then((data) => setProviders(data.providers || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleLink = async (providerId: string) => {
    if (providerId === "credentials") return;
    try {
      await signIn(providerId, { redirect: false });
      // Refresh provider list
      const res = await fetch("/api/account/providers");
      const data = await res.json();
      setProviders(data.providers || []);
      toast.success(`Connected to ${providerId}`);
    } catch {
      toast.error("Failed to connect account");
    }
  };

  const handleUnlink = async (providerId: string) => {
    if (providerId === "credentials") return;
    try {
      const res = await fetch(`/api/account/providers`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: providerId }),
      });
      if (!res.ok) throw new Error();
      setProviders((prev) =>
        prev.map((p) => (p.id === providerId ? { ...p, linked: false } : p))
      );
      toast.success(`Disconnected ${providerId}`);
    } catch {
      toast.error("Failed to disconnect account");
    }
  };

  if (loading) {
    return <div className="text-[#8E8E8E] text-sm py-4">Loading connected accounts...</div>;
  }

  return (
    <div className="space-y-4">
      {providers.map((provider) => (
        <div
          key={provider.id}
          className="flex items-center justify-between p-4 rounded-xl bg-[#1A1A1A] border border-[#222]"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-[#222] flex items-center justify-center text-[#888]">
              {PROVIDER_ICONS[provider.icon] || PROVIDER_ICONS.mail}
            </div>
            <div>
              <p className="text-sm font-semibold">{provider.name}</p>
              <p className="text-xs text-[#666]">
                {provider.linked ? "Connected" : "Not connected"}
              </p>
            </div>
          </div>
          {provider.id !== "credentials" && (
            <button
              onClick={() => (provider.linked ? handleUnlink(provider.id) : handleLink(provider.id))}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition cursor-pointer ${
                provider.linked
                  ? "bg-red-500/10 text-red-400 hover:bg-red-500/20"
                  : "bg-[#34D399]/10 text-[#34D399] hover:bg-[#34D399]/20"
              }`}
            >
              {provider.linked ? (
                <><Unlink className="w-3.5 h-3.5" /> Disconnect</>
              ) : (
                <><Link2 className="w-3.5 h-3.5" /> Connect</>
              )}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Add DELETE handler to `app/api/account/providers/route.ts`**

Append to the existing file:
```typescript
export async function DELETE(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { provider } = await req.json();
  if (!provider) {
    return NextResponse.json({ error: "Provider is required" }, { status: 400 });
  }

  const account = await prisma.account.findFirst({
    where: { userId: session.user.id, provider },
  });
  if (!account) {
    return NextResponse.json({ error: "Account not found" }, { status: 404 });
  }

  await prisma.account.delete({ where: { id: account.id } });

  return NextResponse.json({ success: true });
}
```

Also add the imports at the top (add `prisma` import if not there):
```typescript
import { prisma } from "@/lib/prisma";
```

- [ ] **Step 3: Update `app/account/page.tsx` — replace accounts placeholder**

Replace:
```typescript
          {activeTab === "accounts" && (
            <p className="text-[#8E8E8E] text-sm">Connected accounts section coming in next task.</p>
          )}
```

With:
```typescript
          {activeTab === "accounts" && <ConnectedAccountsSection />}
```

Add import:
```typescript
import ConnectedAccountsSection from "./ConnectedAccountsSection";
```

- [ ] **Step 4: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add app/account/ConnectedAccountsSection.tsx app/api/account/providers/route.ts app/account/page.tsx
git commit -m "feat: add connected accounts section with link/unlink"
```

---

### Task 8: Account page — Preferences Section (theme, language, notifications)

**Files:**
- Create: `app/account/PreferencesSection.tsx`
- Modify: `app/account/page.tsx` (replace placeholder)

- [ ] **Step 1: Create `app/account/PreferencesSection.tsx`**

```typescript
"use client";
import React, { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { Sun, Moon, Globe, Bell, BellOff } from "lucide-react";

interface PreferencesSectionProps {
  preferences: { theme: string; language: string; emailNotifications: boolean };
  onUpdated: (prefs: { theme: string; language: string; emailNotifications: boolean }) => void;
}

export default function PreferencesSection({ preferences, onUpdated }: PreferencesSectionProps) {
  const [theme, setTheme] = useState(preferences.theme);
  const [language, setLanguage] = useState(preferences.language);
  const [emailNotifications, setEmailNotifications] = useState(preferences.emailNotifications);
  const [saving, setSaving] = useState<string | null>(null);

  // Apply theme on mount
  useEffect(() => {
    const stored = localStorage.getItem("harmoniq-theme");
    if (stored) {
      document.documentElement.classList.toggle("dark", stored === "DARK");
      setTheme(stored);
    }
  }, []);

  const savePreference = useCallback(async (key: string, value: any) => {
    setSaving(key);
    try {
      const res = await fetch("/api/account/preferences", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to save preference");
        return;
      }
      onUpdated(data.preferences);
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(null);
    }
  }, [onUpdated]);

  const handleThemeChange = (newTheme: string) => {
    setTheme(newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "DARK");
    localStorage.setItem("harmoniq-theme", newTheme);
    savePreference("theme", newTheme);
  };

  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage);
    savePreference("language", newLanguage);
    toast.success(`Language preference saved (UI translations coming soon)`);
  };

  const handleNotificationChange = (value: boolean) => {
    setEmailNotifications(value);
    savePreference("emailNotifications", value);
  };

  return (
    <div className="space-y-8">
      {/* Theme */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Theme</p>
        <div className="flex gap-3">
          <button
            onClick={() => handleThemeChange("DARK")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              theme === "DARK"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Moon className="w-5 h-5" />
            <span className="text-sm font-semibold">Dark</span>
          </button>
          <button
            onClick={() => handleThemeChange("LIGHT")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              theme === "LIGHT"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Sun className="w-5 h-5" />
            <span className="text-sm font-semibold">Light</span>
          </button>
        </div>
      </div>

      {/* Language */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Language</p>
        <div className="flex gap-3">
          <button
            onClick={() => handleLanguageChange("TH")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              language === "TH"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Globe className="w-5 h-5" />
            <span className="text-sm font-semibold">ไทย</span>
          </button>
          <button
            onClick={() => handleLanguageChange("EN")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              language === "EN"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Globe className="w-5 h-5" />
            <span className="text-sm font-semibold">English</span>
          </button>
        </div>
      </div>

      {/* Email Notifications */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Notifications</p>
        <button
          onClick={() => handleNotificationChange(!emailNotifications)}
          className={`flex items-center gap-3 px-5 py-3 rounded-xl border transition cursor-pointer w-full sm:w-auto ${
            emailNotifications
              ? "bg-[#1A1A1A] border-[#34D399]/40 text-[#34D399]"
              : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
          }`}
        >
          {emailNotifications ? <Bell className="w-5 h-5" /> : <BellOff className="w-5 h-5" />}
          <div className="text-left">
            <p className="text-sm font-semibold">Email Notifications</p>
            <p className="text-xs text-[#666]">
              {emailNotifications ? "Receive emails when processing completes" : "No email notifications"}
            </p>
          </div>
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Update `app/account/page.tsx` — replace preferences placeholder**

Replace:
```typescript
          {activeTab === "preferences" && (
            <p className="text-[#8E8E8E] text-sm">Preferences section coming in next task.</p>
          )}
```

With:
```typescript
          {activeTab === "preferences" && (
            <PreferencesSection
              preferences={data.preferences || { theme: "DARK", language: "TH", emailNotifications: true }}
              onUpdated={(prefs) => setData((prev: any) => ({ ...prev, preferences: prefs }))}
            />
          )}
```

Add import:
```typescript
import PreferencesSection from "./PreferencesSection";
```

- [ ] **Step 3: Run type check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add app/account/PreferencesSection.tsx app/account/page.tsx
git commit -m "feat: add preferences section with theme, language, notifications"
```

---

### Task 9: Tests for new API routes

**Files:**
- Modify: `tests/account.test.ts`

- [ ] **Step 1: Update `tests/account.test.ts` with tests for all new endpoints**

```typescript
jest.mock("@auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

jest.mock("bcryptjs", () => ({
  compare: jest.fn(),
  hash: jest.fn(),
}));

import { GET } from "../app/api/account/route";
import { PUT as PutProfile } from "../app/api/account/profile/route";
import { PUT as PutPassword } from "../app/api/account/password/route";
import { GET as GetProviders } from "../app/api/account/providers/route";
import { PUT as PutPreferences } from "../app/api/account/preferences/route";
import { getServerSession } from "next-auth";
import bcrypt from "bcryptjs";

describe("Account API — GET /api/account", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GET();
    expect(res.status).toBe(401);
  });
});

describe("Account API — PUT /api/account/profile", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ name: "Test" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(401);
  });
});

describe("Account API — PUT /api/account/password", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({ currentPassword: "old", newPassword: "new123" }),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if passwords missing", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "test@test.com" },
    });
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({}),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(400);
  });

  it("should return 400 if new password too short", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "test@test.com" },
    });
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({ currentPassword: "old", newPassword: "ab" }),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(400);
  });
});

describe("Account API — GET /api/account/providers", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GetProviders();
    expect(res.status).toBe(401);
  });
});

describe("Account API — PUT /api/account/preferences", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/preferences", {
      method: "PUT",
      body: JSON.stringify({ theme: "DARK" }),
    });
    const res = await PutPreferences(req);
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run tests**

```bash
npx jest tests/account.test.ts --verbose
```

Expected: All tests pass (7 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/account.test.ts
git commit -m "test: add account API route tests for profile, password, providers, preferences"
```

---

## Verification Checklist

Before marking complete:
- [ ] `npx tsc --noEmit` passes
- [ ] `npx jest` passes (all tests green)
- [ ] `/account` page loads with tabbed layout
- [ ] Profile tab: can edit name, email, avatar URL
- [ ] Password tab: can change password (for credential users)
- [ ] Connected Accounts tab: shows providers, can link/unlink
- [ ] Preferences tab: theme toggle works (dark/light), language/notifications save to DB
- [ ] Theme change persists across page reload (localStorage fallback)
- [ ] All commits are made
