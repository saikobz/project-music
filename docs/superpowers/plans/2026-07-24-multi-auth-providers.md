# Multi-Auth Providers (Credentials, Facebook, LINE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand HarmoniQ NextAuth authentication options to support Email & Password (CredentialsProvider with bcryptjs), Facebook Login, and LINE Login alongside Google.

**Architecture:** Update Prisma User schema to store optional hashed passwords. Configure NextAuth providers (Credentials, Facebook, LINE, Google) in `app/api/auth/[...nextauth]/route.ts`. Create a custom Auth Modal / Page (`app/auth/signin/page.tsx`) with tabs for Login and Register.

**Tech Stack:** Next.js (App Router), React 19, TypeScript, NextAuth.js v5, `bcryptjs`, Prisma ORM.

## Global Constraints

- Python 3.10 compatibility for backend.
- Use `bcryptjs` for secure password hashing (salt rounds: 10).
- Run `npx tsc --noEmit` and unit tests before declaring completion.

---

### Task 1: Prisma Schema Password Field & bcryptjs Installation

**Files:**
- Modify: `prisma/schema.prisma`
- Test: `tests/credentials-db.test.ts`

**Interfaces:**
- Consumes: `bcryptjs`
- Produces: `User.password` field in database schema

- [ ] **Step 1: Write failing test**

Create `tests/credentials-db.test.ts`:
```typescript
import bcrypt from "bcryptjs";
import { prisma } from "../lib/prisma";

describe("Credentials Auth Schema Test", () => {
  it("should create user with hashed password", async () => {
    const hashedPassword = await bcrypt.hash("password123", 10);
    const user = await prisma.user.create({
      data: {
        email: `cred-${Date.now()}@example.com`,
        name: "Cred User",
        password: hashedPassword,
      },
    });

    expect(user.password).toBeDefined();
    const isMatch = await bcrypt.compare("password123", user.password!);
    expect(isMatch).toBe(true);

    await prisma.user.delete({ where: { id: user.id } });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/credentials-db.test.ts`
Expected: FAIL (Cannot find module 'bcryptjs')

- [ ] **Step 3: Install bcryptjs & update schema**

Run: `npm i bcryptjs` and `npm i -D @types/bcryptjs`

Modify `prisma/schema.prisma`:
```prisma
model User {
  id               String        @id @default(cuid())
  name             String?
  email            String        @unique
  password         String?       // Hashed password for Credentials provider
  image            String?
  omiseCustomerId  String?
  subscription     Subscription?
  usageQuotas      UsageQuota[]
  createdAt        DateTime      @default(now())
  updatedAt        DateTime      @updatedAt
}
```

Run: `npx prisma db push`

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/credentials-db.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prisma/schema.prisma tests/credentials-db.test.ts package.json package-lock.json
git commit -m "feat: add password field to Prisma User schema and install bcryptjs"
```

---

### Task 2: NextAuth Configuration for Credentials, Facebook, and LINE Providers

**Files:**
- Create: `app/api/auth/register/route.ts`
- Modify: `app/api/auth/[...nextauth]/route.ts`
- Test: `tests/multi-auth.test.ts`

**Interfaces:**
- Consumes: NextAuth providers (`CredentialsProvider`, `FacebookProvider`, `LineProvider`, `GoogleProvider`), `bcryptjs`
- Produces: Registration API endpoint and multi-provider auth route

- [ ] **Step 1: Write Registration API endpoint**

Create `app/api/auth/register/route.ts`:
```typescript
import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
  try {
    const { name, email, password } = await req.json();

    if (!email || !password) {
      return NextResponse.json({ error: "Email and password are required" }, { status: 400 });
    }

    const existingUser = await prisma.user.findUnique({ where: { email } });
    if (existingUser) {
      return NextResponse.json({ error: "Email already registered" }, { status: 400 });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = await prisma.user.create({
      data: {
        name,
        email,
        password: hashedPassword,
        subscription: { create: { tier: "FREE", status: "ACTIVE" } },
        usageQuotas: {
          create: {
            monthlyQuota: 3,
            usedCount: 0,
            periodStart: new Date(),
            periodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
          },
        },
      },
    });

    return NextResponse.json({ success: true, userId: user.id });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || "Registration failed" }, { status: 500 });
  }
}
```

- [ ] **Step 2: Update NextAuth Providers in route.ts**

Modify `app/api/auth/[...nextauth]/route.ts`:
```typescript
import NextAuth, { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import FacebookProvider from "next-auth/providers/facebook";
import LineProvider from "next-auth/providers/line";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "@/lib/prisma";

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma) as any,
  session: { strategy: "jwt" },
  pages: {
    signIn: "/auth/signin",
  },
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error("Invalid credentials");
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        if (!user || !user.password) {
          throw new Error("No user found with this email");
        }

        const isValid = await bcrypt.compare(credentials.password, user.password);
        if (!isValid) {
          throw new Error("Incorrect password");
        }

        return { id: user.id, email: user.email, name: user.name, image: user.image };
      },
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "google_placeholder",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "google_placeholder",
    }),
    FacebookProvider({
      clientId: process.env.FACEBOOK_CLIENT_ID || "facebook_placeholder",
      clientSecret: process.env.FACEBOOK_CLIENT_SECRET || "facebook_placeholder",
    }),
    LineProvider({
      clientId: process.env.LINE_CLIENT_ID || "line_placeholder",
      clientSecret: process.env.LINE_CLIENT_SECRET || "line_placeholder",
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        const dbUser = await prisma.user.findUnique({
          where: { id: user.id },
          include: { subscription: true },
        });
        token.id = user.id;
        token.tier = dbUser?.subscription?.tier || "FREE";
        token.omiseCustomerId = dbUser?.omiseCustomerId || undefined;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
        session.user.tier = (token.tier as string) || "FREE";
        session.user.omiseCustomerId = token.omiseCustomerId as string;
      }
      return session;
    },
  },
};

const handler = NextAuth(authOptions);
export { handler as GET, handler as POST };
```

Create `tests/multi-auth.test.ts`:
```typescript
import { authOptions } from "../app/api/auth/[...nextauth]/route";

describe("Multi-Auth Providers Test", () => {
  it("should configure Credentials, Google, Facebook, and LINE providers", () => {
    const providerIds = authOptions.providers.map((p) => p.id);
    expect(providerIds).toContain("credentials");
    expect(providerIds).toContain("google");
    expect(providerIds).toContain("facebook");
    expect(providerIds).toContain("line");
  });
});
```

- [ ] **Step 3: Run test to verify it passes**

Run: `npx jest tests/multi-auth.test.ts`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/api/auth/register/route.ts app/api/auth/[...nextauth]/route.ts tests/multi-auth.test.ts
git commit -m "feat: add Credentials, Facebook, and LINE auth providers and register endpoint"
```

---

### Task 3: Custom Sign In & Sign Up Page (`app/auth/signin/page.tsx`)

**Files:**
- Create: `app/auth/signin/page.tsx`

**Interfaces:**
- Consumes: NextAuth `signIn()`, Register API (`/api/auth/register`)
- Produces: Dual-tab (Sign In / Register) UI with social buttons for Google, Facebook, and LINE

- [ ] **Step 1: Write Custom Sign In & Sign Up Page**

Create `app/auth/signin/page.tsx`:
```tsx
"use client";
import React, { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Navbar } from "@/app/components/Navbar";
import { Footer } from "@/app/components/Footer";

export default function SignInPage() {
  const router = useRouter();
  const [tab, setTab] = useState<"signin" | "register">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (tab === "register") {
      try {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Registration failed");

        // Automatically sign in after registration
        const result = await signIn("credentials", { email, password, redirect: false });
        if (result?.error) throw new Error(result.error);
        router.push("/");
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    } else {
      const result = await signIn("credentials", { email, password, redirect: false });
      setLoading(false);
      if (result?.error) {
        setError("Invalid email or password");
      } else {
        router.push("/");
      }
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow flex items-center justify-center p-4">
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-8 max-w-md w-full shadow-2xl space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-2xl font-bold tracking-tight">Welcome to HarmoniQ</h1>
            <p className="text-xs text-[#8E8E8E]">AI Music Separator & Audio Toolkit</p>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-[#222222]">
            <button
              onClick={() => setTab("signin")}
              className={`flex-1 py-2 text-sm font-semibold transition border-b-2 ${
                tab === "signin"
                  ? "border-[#34D399] text-[#34D399]"
                  : "border-transparent text-[#8E8E8E] hover:text-[#F3F3F3]"
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => setTab("register")}
              className={`flex-1 py-2 text-sm font-semibold transition border-b-2 ${
                tab === "register"
                  ? "border-[#34D399] text-[#34D399]"
                  : "border-transparent text-[#8E8E8E] hover:text-[#F3F3F3]"
              }`}
            >
              Create Account
            </button>
          </div>

          {error && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg">{error}</div>}

          {/* Email & Password Form */}
          <form onSubmit={handleCredentialsSubmit} className="space-y-4">
            {tab === "register" && (
              <div>
                <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Full Name</label>
                <input
                  type="text"
                  required
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="John Doe"
                  className="w-full bg-[#1A1A1A] border border-[#333333] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#34D399]"
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Email Address</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full bg-[#1A1A1A] border border-[#333333] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#34D399]"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Password</label>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-[#1A1A1A] border border-[#333333] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#34D399]"
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] font-bold text-sm rounded-lg transition cursor-pointer"
            >
              {loading ? "Processing..." : tab === "signin" ? "Sign In with Email" : "Create Account"}
            </button>
          </form>

          <div className="relative flex items-center justify-center border-t border-[#222222] pt-4">
            <span className="bg-[#111111] px-2 text-[10px] uppercase text-[#8E8E8E] font-bold">Or continue with</span>
          </div>

          {/* Social Logins */}
          <div className="space-y-2">
            <button
              onClick={() => signIn("google")}
              className="w-full py-2 bg-[#1A1A1A] hover:bg-[#222222] border border-[#333333] text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
            >
              🌐 Google
            </button>
            <button
              onClick={() => signIn("facebook")}
              className="w-full py-2 bg-[#1877F2]/10 hover:bg-[#1877F2]/20 border border-[#1877F2]/40 text-[#1877F2] text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
            >
              📘 Facebook
            </button>
            <button
              onClick={() => signIn("line")}
              className="w-full py-2 bg-[#00C300]/10 hover:bg-[#00C300]/20 border border-[#00C300]/40 text-[#00C300] text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
            >
              💬 LINE Login
            </button>
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
git add app/auth/signin/page.tsx
git commit -m "feat: add custom Sign In and Register page with Email & Password, Facebook, and LINE login options"
```
