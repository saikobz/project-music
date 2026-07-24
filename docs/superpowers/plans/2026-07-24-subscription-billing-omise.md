# Subscription & Billing (Omise + NextAuth + FastAPI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 3-tier subscription system (Free, Basic, Pro) using Omise (Opn Payments), NextAuth.js, Prisma ORM, and FastAPI middleware for HarmoniQ.

**Architecture:** NextAuth.js authenticates users and manages sessions in a Prisma ORM database. Next.js App Router API routes handle Omise checkout and webhooks (Schedule API for Credit Cards, Source Charges for PromptPay QR). FastAPI backend validates session tokens, enforces monthly quotas, and restricts the AutoEQ CNN model for Free tier users.

**Tech Stack:** Next.js (App Router), React 19, TypeScript, Prisma ORM (SQLite/PostgreSQL), NextAuth.js v5, Omise Node.js SDK, FastAPI (Python 3.10), PyTorch.

## Global Constraints

- Python Compatibility: Python 3.10 (Do NOT use Python 3.11+ syntax like `ExceptionGroup` or `typing.Self`).
- Frontend Environment Variables: Prefixed with `NEXT_PUBLIC_` (e.g., `NEXT_PUBLIC_API_BASE`).
- File Upload Limit: Max 100MB WAV files.
- Testing Verification: Run `npx tsc --noEmit` and unit tests before declaring completion.

---

### Task 1: Prisma ORM Database Setup & Schema

**Files:**
- Create: `prisma/schema.prisma`
- Create: `lib/prisma.ts`
- Test: `tests/db.test.ts`

**Interfaces:**
- Consumes: Database connection string (`DATABASE_URL`)
- Produces: Prisma Client instance exporting `User`, `Subscription`, `UsageQuota`

- [ ] **Step 1: Write the failing test**

Create `tests/db.test.ts`:
```typescript
import { prisma } from "../lib/prisma";

describe("Database Schema Test", () => {
  it("should create user with default FREE subscription tier", async () => {
    const user = await prisma.user.create({
      data: {
        email: "test@example.com",
        name: "Test User",
        subscription: {
          create: {
            tier: "FREE",
            status: "ACTIVE",
          },
        },
      },
      include: { subscription: true },
    });

    expect(user.email).toBe("test@example.com");
    expect(user.subscription?.tier).toBe("FREE");

    // Clean up
    await prisma.user.delete({ where: { id: user.id } });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/db.test.ts`
Expected: FAIL with "Cannot find module '../lib/prisma'"

- [ ] **Step 3: Write minimal implementation**

Create `prisma/schema.prisma`:
```prisma
datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

generator client {
  provider = "prisma-client-js"
}

model User {
  id               String        @id @default(cuid())
  name             String?
  email            String        @unique
  image            String?
  omiseCustomerId  String?
  subscription     Subscription?
  usageQuotas      UsageQuota[]
  createdAt        DateTime      @default(now())
  updatedAt        DateTime      @updatedAt
}

model Subscription {
  id                 String             @id @default(cuid())
  userId             String             @unique
  user               User               @relation(fields: [userId], references: [id], onDelete: Cascade)
  tier               String             @default("FREE") // FREE, BASIC, PRO
  status             String             @default("ACTIVE") // ACTIVE, PAST_DUE, EXPIRED
  paymentMethod      String?            // CREDIT_CARD, PROMPTPAY
  omiseScheduleId    String?
  currentPeriodStart DateTime           @default(now())
  currentPeriodEnd   DateTime?
  createdAt          DateTime           @default(now())
  updatedAt          DateTime           @updatedAt
}

model UsageQuota {
  id           String   @id @default(cuid())
  userId       String
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)
  monthlyQuota Int      // FREE=1, BASIC=15, PRO=-1
  usedCount    Int      @default(0)
  periodStart  DateTime
  periodEnd    DateTime
}
```

Create `lib/prisma.ts`:
```typescript
import { PrismaClient } from "@prisma/client";

const globalForPrisma = global as unknown as { prisma: PrismaClient };

export const prisma = globalForPrisma.prisma || new PrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
```

Run Prisma migration: `npx prisma db push`

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/db.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add prisma/schema.prisma lib/prisma.ts tests/db.test.ts
git commit -m "feat: setup Prisma ORM schema for users and subscriptions"
```

---

### Task 2: NextAuth.js Integration & Session Enrichment

**Files:**
- Create: `app/api/auth/[...nextauth]/route.ts`
- Create: `types/next-auth.d.ts`
- Test: `tests/auth.test.ts`

**Interfaces:**
- Consumes: Prisma Client (`prisma`)
- Produces: JWT Tokens & Session containing `user.id`, `user.tier`, `user.omiseCustomerId`

- [ ] **Step 1: Write failing test for session enrichment**

Create `tests/auth.test.ts`:
```typescript
import { authOptions } from "../app/api/auth/[...nextauth]/route";

describe("NextAuth Session Callback", () => {
  it("should enrich session with user tier and subscription status", async () => {
    const mockSession = { user: { email: "test@example.com" } };
    const mockToken = { sub: "user-123", tier: "PRO" };

    const session = await authOptions.callbacks.session({
      session: mockSession,
      token: mockToken,
    });

    expect(session.user.tier).toBe("PRO");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/auth.test.ts`
Expected: FAIL with "Cannot find module"

- [ ] **Step 3: Write minimal implementation**

Create `types/next-auth.d.ts`:
```typescript
import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      tier: string;
      omiseCustomerId?: string;
    } & DefaultSession["user"];
  }

  interface JWT {
    id: string;
    tier: string;
    omiseCustomerId?: string;
  }
}
```

Create `app/api/auth/[...nextauth]/route.ts`:
```typescript
import NextAuth, { NextAuthOptions } from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import CredentialsProvider from "next-auth/providers/credentials";
import { PrismaAdapter } from "@nextauth/prisma-adapter";
import { prisma } from "@/lib/prisma";

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  session: { strategy: "jwt" },
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
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

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/auth.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/auth/[...nextauth]/route.ts types/next-auth.d.ts tests/auth.test.ts
git commit -m "feat: implement NextAuth.js authentication and session enrichment"
```

---

### Task 3: Omise SDK Integration & Checkout API Endpoint

**Files:**
- Create: `lib/omise.ts`
- Create: `app/api/subscription/checkout/route.ts`
- Test: `tests/checkout.test.ts`

**Interfaces:**
- Consumes: Omise Secret Key (`OMISE_SECRET_KEY`), `prisma`
- Produces: POST Endpoint `/api/subscription/checkout`

- [ ] **Step 1: Write failing test**

Create `tests/checkout.test.ts`:
```typescript
import { POST } from "../app/api/subscription/checkout/route";

describe("Checkout API Test", () => {
  it("should return 401 if user is unauthenticated", async () => {
    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({ tier: "BASIC", paymentMethod: "PROMPTPAY" }),
    });

    const res = await POST(req);
    expect(res.status).toBe(401);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/checkout.test.ts`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create `lib/omise.ts`:
```typescript
import Omise from "omise";

export const omise = Omise({
  publicKey: process.env.NEXT_PUBLIC_OMISE_PUBLIC_KEY || "",
  secretKey: process.env.OMISE_SECRET_KEY || "",
});
```

Create `app/api/subscription/checkout/route.ts`:
```typescript
import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/app/api/auth/[...nextauth]/route";
import { omise } from "@/lib/omise";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { tier, paymentMethod, cardToken } = await req.json();
  const amount = tier === "PRO" ? 29900 : 9900; // In Satang (THB * 100)

  try {
    if (paymentMethod === "PROMPTPAY") {
      const charge = await omise.charges.create({
        amount,
        currency: "thb",
        source: { type: "promptpay" },
        metadata: { userId: session.user.id, tier },
      });

      return NextResponse.json({
        success: true,
        chargeId: charge.id,
        qrCodeUrl: charge.source.scannable_code.image.download_uri,
      });
    }

    if (paymentMethod === "CREDIT_CARD" && cardToken) {
      let customerId = session.user.omiseCustomerId;
      if (!customerId) {
        const customer = await omise.customers.create({
          email: session.user.email || "",
          card: cardToken,
        });
        customerId = customer.id;
        await prisma.user.update({
          where: { id: session.user.id },
          data: { omiseCustomerId: customerId },
        });
      }

      const schedule = await omise.schedules.create({
        every: 1,
        period: "month",
        charge: { customer: customerId, amount, currency: "thb" },
      });

      await prisma.subscription.upsert({
        where: { userId: session.user.id },
        update: { tier, status: "ACTIVE", paymentMethod, omiseScheduleId: schedule.id },
        create: { userId: session.user.id, tier, status: "ACTIVE", paymentMethod, omiseScheduleId: schedule.id },
      });

      return NextResponse.json({ success: true, scheduleId: schedule.id });
    }

    return NextResponse.json({ error: "Invalid payment method" }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/checkout.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/omise.ts app/api/subscription/checkout/route.ts tests/checkout.test.ts
git commit -m "feat: implement Omise checkout API for Card and PromptPay"
```

---

### Task 4: Omise Webhook Handler Endpoint

**Files:**
- Create: `app/api/webhooks/omise/route.ts`
- Test: `tests/webhook.test.ts`

**Interfaces:**
- Consumes: Omise Event Payloads (`charge.complete`, `schedule.process`)
- Produces: POST Endpoint `/api/webhooks/omise`

- [ ] **Step 1: Write failing test**

Create `tests/webhook.test.ts`:
```typescript
import { POST } from "../app/api/webhooks/omise/route";

describe("Omise Webhook Test", () => {
  it("should process charge.complete event for PromptPay", async () => {
    const payload = {
      key: "charge.complete",
      data: {
        id: "chrg_test_123",
        status: "successful",
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };

    const req = new Request("http://localhost:3000/api/webhooks/omise", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/webhook.test.ts`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Create `app/api/webhooks/omise/route.ts`:
```typescript
import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { key, data } = body;

    if (key === "charge.complete" && data.status === "successful") {
      const userId = data.metadata?.userId;
      const tier = data.metadata?.tier;

      if (userId && tier) {
        const periodEnd = new Date();
        periodEnd.setDate(periodEnd.getDate() + 30);

        await prisma.subscription.upsert({
          where: { userId },
          update: { tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
          create: { userId, tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
        });

        // Reset Quota
        const maxQuota = tier === "PRO" ? -1 : tier === "BASIC" ? 15 : 1;
        await prisma.usageQuota.create({
          data: {
            userId,
            monthlyQuota: maxQuota,
            usedCount: 0,
            periodStart: new Date(),
            periodEnd,
          },
        });
      }
    }

    return NextResponse.json({ received: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/webhook.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/api/webhooks/omise/route.ts tests/webhook.test.ts
git commit -m "feat: implement Omise Webhook handler for updating subscription status"
```

---

### Task 5: FastAPI Quota & Model Permission Guard

**Files:**
- Create: `backend/utils/auth_guard.py`
- Modify: `backend/main.py:100-150`
- Test: `backend/tests/test_auth_guard.py`

**Interfaces:**
- Consumes: HTTP Headers `Authorization: Bearer <jwt_token>`
- Produces: Middleware guard checking `tier` and `model_type`

- [ ] **Step 1: Write failing test in Python**

Create `backend/tests/test_auth_guard.py`:
```python
import pytest
from backend.utils.auth_guard import validate_tier_and_quota

def test_free_tier_cannot_use_cnn_model():
    with pytest.raises(Exception) as exc_info:
        validate_tier_and_quota(user_tier="FREE", used_quota=0, model_type="CNN")
    assert "CNN model requires Basic or Pro tier" in str(exc_info.value)

def test_free_tier_exceeded_quota():
    with pytest.raises(Exception) as exc_info:
        validate_tier_and_quota(user_tier="FREE", used_quota=1, model_type="LSTM")
    assert "Monthly quota reached" in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_auth_guard.py`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal Python implementation**

Create `backend/utils/auth_guard.py`:
```python
from fastapi import HTTPException, status

def validate_tier_and_quota(user_tier: str, used_quota: int, model_type: str = "LSTM"):
    """
    Validates user tier permissions and usage quotas for music separation and AutoEQ.
    Python 3.10 compatible.
    """
    # 1. Model Lock Check
    if model_type.upper() == "CNN" and user_tier.upper() == "FREE":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="AutoEQ CNN model requires Basic or Pro subscription. Please upgrade to unlock."
        )

    # 2. Quota Check
    tier_limits = {
        "FREE": 1,
        "BASIC": 15,
        "PRO": -1  # Unlimited
    }

    limit = tier_limits.get(user_tier.upper(), 1)
    if limit != -1 and used_quota >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Monthly quota reached for {user_tier} tier ({used_quota}/{limit}). Please upgrade for more processing."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_auth_guard.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/utils/auth_guard.py backend/tests/test_auth_guard.py
git commit -m "feat: add FastAPI Auth Guard for quota enforcement and CNN model locking"
```

---

### Task 6: Pricing Page & Checkout UI Component

**Files:**
- Create: `app/pricing/page.tsx`
- Create: `app/components/CheckoutModal.tsx`

**Interfaces:**
- Consumes: NextAuth session (`useSession()`), Checkout API (`/api/subscription/checkout`)
- Produces: Interactive Pricing UI with PromptPay QR display modal

- [ ] **Step 1: Write component code**

Create `app/components/CheckoutModal.tsx`:
```tsx
"use client";
import { useState } from "react";

interface CheckoutModalProps {
  isOpen: boolean;
  onClose: () => void;
  tier: "BASIC" | "PRO";
  price: number;
}

export default function CheckoutModal({ isOpen, onClose, tier, price }: CheckoutModalProps) {
  const [paymentMethod, setPaymentMethod] = useState<"PROMPTPAY" | "CREDIT_CARD">("PROMPTPAY");
  const [qrUrl, setQrUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleCheckout = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/subscription/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tier, paymentMethod }),
      });
      const data = await res.json();
      if (data.qrCodeUrl) {
        setQrUrl(data.qrCodeUrl);
      } else if (data.success) {
        alert("Subscription activated successfully!");
        onClose();
      }
    } catch (err) {
      alert("Payment failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-md w-full text-white">
        <h2 className="text-xl font-bold mb-2">Subscribe to {tier} Tier</h2>
        <p className="text-slate-400 text-sm mb-4">Total Amount: {price} THB/month</p>

        {qrUrl ? (
          <div className="text-center py-4">
            <p className="text-sm font-semibold mb-2">Scan QR Code via Mobile Banking App</p>
            <img src={qrUrl} alt="PromptPay QR Code" className="mx-auto w-64 h-64 rounded-lg bg-white p-2" />
            <button onClick={onClose} className="mt-4 px-4 py-2 bg-slate-800 text-slate-300 rounded-lg">Done</button>
          </div>
        ) : (
          <div>
            <div className="space-y-2 mb-6">
              <label className="flex items-center space-x-3 p-3 bg-slate-800 rounded-lg cursor-pointer">
                <input
                  type="radio"
                  name="payment"
                  checked={paymentMethod === "PROMPTPAY"}
                  onChange={() => setPaymentMethod("PROMPTPAY")}
                />
                <span>PromptPay QR Code</span>
              </label>
              <label className="flex items-center space-x-3 p-3 bg-slate-800 rounded-lg cursor-pointer">
                <input
                  type="radio"
                  name="payment"
                  checked={paymentMethod === "CREDIT_CARD"}
                  onChange={() => setPaymentMethod("CREDIT_CARD")}
                />
                <span>Credit / Debit Card</span>
              </label>
            </div>
            <div className="flex justify-end space-x-2">
              <button onClick={onClose} className="px-4 py-2 bg-slate-800 rounded-lg">Cancel</button>
              <button
                onClick={handleCheckout}
                disabled={loading}
                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-semibold"
              >
                {loading ? "Processing..." : "Pay Now"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

Create `app/pricing/page.tsx`:
```tsx
"use client";
import { useState } from "react";
import CheckoutModal from "../components/CheckoutModal";

export default function PricingPage() {
  const [selectedTier, setSelectedTier] = useState<"BASIC" | "PRO" | null>(null);

  return (
    <div className="min-h-screen bg-slate-950 text-white py-16 px-4">
      <div className="max-w-5xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-extrabold mb-4 bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent">
          HarmoniQ Plans & Pricing
        </h1>
        <p className="text-slate-400">Choose the perfect plan for your music separation & mastering needs</p>
      </div>

      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {/* Free Plan */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-xl font-bold mb-2">Free</h3>
            <div className="text-3xl font-extrabold mb-4">0 THB</div>
            <ul className="text-slate-400 text-sm space-y-2 mb-6">
              <li>• 1 song / month</li>
              <li>• AutoEQ LSTM Model</li>
              <li>• 🔒 CNN Model Locked</li>
            </ul>
          </div>
          <button className="w-full py-2 bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed">Current Plan</button>
        </div>

        {/* Basic Plan */}
        <div className="bg-slate-900 border border-indigo-500/40 rounded-2xl p-6 flex flex-col justify-between relative">
          <div>
            <h3 className="text-xl font-bold mb-2">Basic</h3>
            <div className="text-3xl font-extrabold mb-4">99 THB<span className="text-sm font-normal text-slate-400">/mo</span></div>
            <ul className="text-slate-400 text-sm space-y-2 mb-6">
              <li>• 15 songs / month</li>
              <li>• AutoEQ LSTM & CNN Models</li>
              <li>• Lossless WAV Export</li>
            </ul>
          </div>
          <button
            onClick={() => setSelectedTier("BASIC")}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 font-semibold text-white rounded-lg"
          >
            Upgrade to Basic
          </button>
        </div>

        {/* Pro Plan */}
        <div className="bg-gradient-to-b from-indigo-900/40 to-slate-900 border border-indigo-400 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-xl font-bold mb-2 text-indigo-300">Pro</h3>
            <div className="text-3xl font-extrabold mb-4">299 THB<span className="text-sm font-normal text-slate-400">/mo</span></div>
            <ul className="text-slate-300 text-sm space-y-2 mb-6">
              <li>• Unlimited songs / month</li>
              <li>• All AutoEQ Models (LSTM & CNN)</li>
              <li>• Full AI Auto Mastering</li>
            </ul>
          </div>
          <button
            onClick={() => setSelectedTier("PRO")}
            className="w-full py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:opacity-90 font-semibold text-white rounded-lg"
          >
            Upgrade to Pro
          </button>
        </div>
      </div>

      {selectedTier && (
        <CheckoutModal
          isOpen={!!selectedTier}
          onClose={() => setSelectedTier(null)}
          tier={selectedTier}
          price={selectedTier === "PRO" ? 299 : 99}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript & Build check**

Run: `npx tsc --noEmit`
Expected: PASS (No compilation errors)

- [ ] **Step 3: Commit**

```bash
git add app/pricing/page.tsx app/components/CheckoutModal.tsx
git commit -m "feat: add Pricing Page and Checkout Modal for Omise payments"
```
