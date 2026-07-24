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

  const body = await req.json();
  const { tier, paymentMethod, cardToken } = body;
  const amount = tier === "PRO" ? 29900 : 9900; // Amount in Satang (THB * 100)

  try {
    if (paymentMethod === "PROMPTPAY") {
      const charge = await omise.charges.create({
        amount,
        currency: "thb",
        source: { type: "promptpay" },
        metadata: { userId: session.user.id, tier },
      });

      const qrCodeUrl = (charge.source as any)?.scannable_code?.image?.download_uri || null;

      return NextResponse.json({
        success: true,
        chargeId: charge.id,
        qrCodeUrl,
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
    return NextResponse.json({ error: error.message || "Checkout failed" }, { status: 500 });
  }
}
