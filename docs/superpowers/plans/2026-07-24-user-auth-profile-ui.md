# User Authentication & Profile UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete User Authentication UI & Profile Management system including NextAuth SessionProvider, User Menu Dropdown on Navbar, and Account Dashboard page (`/account`) for HarmoniQ.

**Architecture:** Wrap the Next.js application layout in NextAuth `SessionProvider`. Build a responsive `UserMenu` component for the `Navbar` to handle Sign In / Sign Out and display tier badges. Create an `/account` page showing profile info, active subscription status, and usage quota bars.

**Tech Stack:** Next.js (App Router), React 19, TypeScript, NextAuth.js v5, Tailwind CSS 4, Lucide React / SVG Icons, Prisma ORM.

## Global Constraints

- Follow HarmoniQ Dark Aesthetic (`#0A0A0A` background, `#111111` card surfaces, `#34D399` accent green, `#F3F3F3` text).
- Use `useSession` hook for client-side state management.
- Run `npx tsc --noEmit` and unit tests before declaring completion.

---

### Task 1: NextAuth SessionProvider & Root Layout Wrapper

**Files:**
- Create: `app/components/SessionProvider.tsx`
- Modify: `app/layout.tsx`
- Test: `tests/session.test.ts`

**Interfaces:**
- Consumes: `SessionProvider` from `next-auth/react`
- Produces: `<AuthProvider>` wrapper around root layout

- [ ] **Step 1: Write the failing test**

Create `tests/session.test.ts`:
```typescript
import AuthProvider from "../app/components/SessionProvider";

describe("AuthProvider Component", () => {
  it("should be defined as a React component", () => {
    expect(AuthProvider).toBeDefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/session.test.ts`
Expected: FAIL with "Cannot find module"

- [ ] **Step 3: Write minimal implementation**

Create `app/components/SessionProvider.tsx`:
```tsx
"use client";
import React from "react";
import { SessionProvider as NextAuthSessionProvider } from "next-auth/react";

export default function AuthProvider({ children }: { children: React.ReactNode }) {
  return <NextAuthSessionProvider>{children}</NextAuthSessionProvider>;
}
```

Modify `app/layout.tsx`:
```tsx
import type { Metadata } from "next";
import AuthProvider from "./components/SessionProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "HarmoniQ — AI Audio Toolkit",
  description: "ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="th" className="dark">
      <body className="bg-[#0A0A0A] text-[#F3F3F3] antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/session.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/components/SessionProvider.tsx app/layout.tsx tests/session.test.ts
git commit -m "feat: wrap root layout with NextAuth SessionProvider"
```

---

### Task 2: User Menu & Profile Dropdown Component for Navbar

**Files:**
- Create: `app/components/UserMenu.tsx`
- Modify: `app/components/Navbar.tsx`
- Test: `tests/user-menu.test.ts`

**Interfaces:**
- Consumes: `useSession()`, `signIn()`, `signOut()` from `next-auth/react`
- Produces: `<UserMenu />` component integrated into `Navbar`

- [ ] **Step 1: Write component and unit test**

Create `tests/user-menu.test.ts`:
```typescript
import UserMenu from "../app/components/UserMenu";

describe("UserMenu Component", () => {
  it("should be defined", () => {
    expect(UserMenu).toBeDefined();
  });
});
```

Create `app/components/UserMenu.tsx`:
```tsx
"use client";
import React, { useState } from "react";
import Link from "next/link";
import { useSession, signIn, signOut } from "next-auth/react";

export default function UserMenu() {
  const { data: session, status } = useSession();
  const [open, setOpen] = useState(false);

  if (status === "loading") {
    return <div className="h-8 w-20 bg-[#1A1A1A] animate-pulse rounded-md"></div>;
  }

  if (!session || !session.user) {
    return (
      <button
        onClick={() => signIn()}
        className="px-4 py-1.5 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] text-sm font-semibold rounded-md transition cursor-pointer"
      >
        Sign In
      </button>
    );
  }

  const user = session.user;
  const tier = (user as any).tier || "FREE";

  const tierColors: Record<string, string> = {
    FREE: "bg-slate-800 text-slate-300 border-slate-700",
    BASIC: "bg-[#34D399]/10 text-[#34D399] border-[#34D399]/30",
    PRO: "bg-purple-950/40 text-purple-400 border-purple-500/30",
  };

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 p-1 rounded-full border border-[#222222] bg-[#111111] hover:border-[#333333] transition cursor-pointer"
      >
        {user.image ? (
          <img src={user.image} alt={user.name || "User Avatar"} className="w-8 h-8 rounded-full" />
        ) : (
          <div className="w-8 h-8 rounded-full bg-[#222222] flex items-center justify-center text-xs font-bold text-[#F3F3F3]">
            {user.name ? user.name[0].toUpperCase() : "U"}
          </div>
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-[#111111] border border-[#222222] rounded-xl shadow-2xl p-2 z-50 space-y-1">
          <div className="px-3 py-2 border-b border-[#222222]">
            <p className="text-sm font-bold text-[#F3F3F3] truncate">{user.name || "User Account"}</p>
            <p className="text-xs text-[#8E8E8E] truncate mb-2">{user.email}</p>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${tierColors[tier] || tierColors.FREE}`}>
              {tier} PLAN
            </span>
          </div>

          <Link
            href="/account"
            onClick={() => setOpen(false)}
            className="block px-3 py-2 text-xs text-[#CCCCCC] hover:text-white hover:bg-[#1A1A1A] rounded-md transition"
          >
            ⚙️ Account Settings
          </Link>
          <Link
            href="/pricing"
            onClick={() => setOpen(false)}
            className="block px-3 py-2 text-xs text-[#34D399] hover:bg-[#1A1A1A] rounded-md transition"
          >
            ⚡ Upgrade Plan
          </Link>

          <button
            onClick={() => signOut()}
            className="w-full text-left px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 rounded-md transition cursor-pointer"
          >
            🚪 Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
```

Modify `app/components/Navbar.tsx`: Integrate `<UserMenu />` next to navigation items.

- [ ] **Step 2: Verify component builds cleanly**

Run: `npx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/components/UserMenu.tsx app/components/Navbar.tsx tests/user-menu.test.ts
git commit -m "feat: add UserMenu dropdown component to Navbar"
```

---

### Task 3: User Account Dashboard Page (`/account`)

**Files:**
- Create: `app/account/page.tsx`
- Create: `app/api/account/route.ts`
- Test: `tests/account.test.ts`

**Interfaces:**
- Consumes: NextAuth session, Prisma User & Subscription records
- Produces: `/account` page with profile info, active tier, quota usage progress bar, and billing details

- [ ] **Step 1: Write API and Dashboard UI**

Create `app/api/account/route.ts`:
```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { prisma } from "@/lib/prisma";

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
    subscription: user.subscription || { tier: "FREE", status: "ACTIVE" },
    quota: currentQuota,
  });
}
```

Create `app/account/page.tsx`:
```tsx
"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export default function AccountPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/account")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

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
          <p className="text-red-400">Please sign in to view your account details.</p>
          <Link href="/api/auth/signin" className="inline-block px-6 py-2 bg-[#34D399] text-[#0A0A0A] font-bold rounded-lg">
            Sign In
          </Link>
        </div>
        <Footer />
      </div>
    );
  }

  const { user, subscription, quota } = data;
  const used = quota.usedCount || 0;
  const max = quota.monthlyQuota;
  const isUnlimited = max === -1;
  const percent = isUnlimited ? 0 : Math.min(100, Math.round((used / max) * 100));

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow mx-auto w-full max-w-3xl px-4 py-12 space-y-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Account Dashboard</h1>
          <p className="text-[#8E8E8E] text-sm mt-1">Manage your profile, active subscription, and usage quotas.</p>
        </div>

        {/* Profile Card */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 flex items-center gap-4">
          {user.image ? (
            <img src={user.image} alt={user.name} className="w-16 h-16 rounded-full" />
          ) : (
            <div className="w-16 h-16 rounded-full bg-[#222222] flex items-center justify-center text-xl font-bold text-[#F3F3F3]">
              {user.name ? user.name[0].toUpperCase() : "U"}
            </div>
          )}
          <div>
            <h2 className="text-lg font-bold">{user.name || "HarmoniQ User"}</h2>
            <p className="text-sm text-[#8E8E8E]">{user.email}</p>
          </div>
        </div>

        {/* Subscription & Quota Card */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#222222] pb-4">
            <div>
              <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider">Current Subscription</p>
              <h3 className="text-xl font-extrabold text-[#34D399] mt-0.5">{subscription.tier} PLAN</h3>
            </div>
            <Link
              href="/pricing"
              className="px-4 py-2 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] text-xs font-bold rounded-lg transition"
            >
              Change Plan
            </Link>
          </div>

          {/* Quota Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-[#CCCCCC]">Monthly Song Processing Quota</span>
              <span className="font-bold text-[#34D399]">
                {isUnlimited ? "Unlimited" : `${used} / ${max} songs used`}
              </span>
            </div>
            {!isUnlimited && (
              <div className="w-full bg-[#222222] rounded-full h-2.5 overflow-hidden">
                <div className="bg-[#34D399] h-2.5 rounded-full transition-all duration-300" style={{ width: `${percent}%` }}></div>
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
```

- [ ] **Step 2: Run TypeScript check**

Run: `npx tsc --noEmit`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add app/account/page.tsx app/api/account/route.ts
git commit -m "feat: implement User Account Dashboard page and API endpoint"
```
