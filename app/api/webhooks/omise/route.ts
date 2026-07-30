import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import crypto from "crypto";

async function savePaymentRecord(userId: string, chargeId: string, amount: number, status: string) {
  try {
    await prisma.paymentRecord.create({
      data: { userId, omiseChargeId: chargeId, amount, currency: "thb", status },
    });
  } catch (e: any) {
    if (!e?.code || e.code !== "P2002") {
      console.error("Failed to save payment record:", e);
    }
  }
}

async function activateSubscription(userId: string, tier: string) {
  const periodEnd = new Date();
  periodEnd.setDate(periodEnd.getDate() + 30);

  const maxQuota = tier === "PRO" ? -1 : tier === "BASIC" ? 15 : 3;

  await prisma.subscription.upsert({
    where: { userId },
    update: { tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
    create: { userId, tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
  });

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

async function markSubscriptionPastDue(userId: string) {
  await prisma.subscription.update({
    where: { userId },
    data: { status: "PAST_DUE" },
  });
}

async function markSubscriptionExpired(userId: string) {
  await prisma.subscription.update({
    where: { userId },
    data: { status: "EXPIRED" },
  });
}

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { key, data } = body;

    const signature = req.headers.get("x-omise-signature") || req.headers.get("X-Omise-Signature");
    const webhookSecret = process.env.OMISE_WEBHOOK_SECRET;
    if (webhookSecret && signature) {
      const rawBody = JSON.stringify(body);
      const expected = crypto.createHmac("sha256", webhookSecret).update(rawBody).digest("hex");
      if (signature !== expected) {
        return NextResponse.json({ error: "Invalid signature" }, { status: 403 });
      }
    }

    if (key === "charge.complete") {
      const userId = data?.metadata?.userId;
      const tier = data?.metadata?.tier || "BASIC";
      const amount = data?.amount || 0;

      if (userId) {
        await savePaymentRecord(userId, data.id, amount, "successful");
        if (data.status === "successful") {
          await activateSubscription(userId, tier);
        }
      }
    }

    if (key === "charge.failed") {
      const userId = data?.metadata?.userId;
      if (userId) {
        await savePaymentRecord(userId, data.id, data.amount || 0, "failed");
      }
    }

    if (key === "charge.expired") {
      const userId = data?.metadata?.userId;
      if (userId) {
        await savePaymentRecord(userId, data.id, data.amount || 0, "expired");
      }
    }

    if (key === "schedule.process") {
      const userId = data?.metadata?.userId;
      const tier = data?.metadata?.tier || "BASIC";
      const chargeId = data?.charge;
      const amount = data?.amount || 0;

      if (userId) {
        if (data.status === "successful") {
          if (chargeId) await savePaymentRecord(userId, chargeId, amount, "successful");
          await activateSubscription(userId, tier);
        } else if (data.status === "failed") {
          if (chargeId) await savePaymentRecord(userId, chargeId, amount, "failed");
          await markSubscriptionPastDue(userId);
        }
      }
    }

    if (key === "schedule.expired") {
      const userId = data?.metadata?.userId;
      if (userId) {
        await markSubscriptionExpired(userId);
      }
    }

    return NextResponse.json({ received: true });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || "Webhook error" }, { status: 500 });
  }
}
