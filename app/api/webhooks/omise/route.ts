import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { key, data } = body;

    if ((key === "charge.complete" || key === "schedule.process") && data?.status === "successful") {
      const userId = data.metadata?.userId;
      const tier = data.metadata?.tier || "BASIC";

      if (userId) {
        const periodEnd = new Date();
        periodEnd.setDate(periodEnd.getDate() + 30);

        await prisma.subscription.upsert({
          where: { userId },
          update: { tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
          create: { userId, tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
        });

        // Reset Quota (Free=3, Basic=15, Pro=-1)
        const maxQuota = tier === "PRO" ? -1 : tier === "BASIC" ? 15 : 3;
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
    return NextResponse.json({ error: error.message || "Webhook error" }, { status: 500 });
  }
}
