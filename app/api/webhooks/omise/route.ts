import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import crypto from "crypto";
import { omise } from "@/lib/omise";
import { PERIOD_MS } from "@/lib/config";
import { getMonthlyQuota } from "@/lib/subscription";

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
  const periodStart = new Date();
  const periodEnd = new Date(Date.now() + PERIOD_MS);

  const maxQuota = getMonthlyQuota(tier);

  await prisma.subscription.upsert({
    where: { userId },
    update: { tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
    create: { userId, tier, status: "ACTIVE", currentPeriodEnd: periodEnd },
  });

  // L8: ล้าง quota รอบเก่าให้เหลือรอบเดียว (เดิมสร้างแถวใหม่ทุกครั้งที่ renew -> ตารางบานเรื่อยๆ)
  await prisma.usageQuota.deleteMany({ where: { userId } });
  await prisma.usageQuota.create({
    data: {
      userId,
      monthlyQuota: maxQuota,
      usedCount: 0,
      periodStart,
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
  // อ่าน raw body ก่อนเสมอ เพื่อใช้ยืนยันลายเซ็นกับข้อความต้นฉบับ
  // (ห้ามใช้ JSON.stringify(body) ใหม่ เพราะ format อาจต่างจากที่ Omise เซ็น)
  let rawBody: string;
  try {
    rawBody = await req.text();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  // ยืนยันความถูกต้องของ webhook (G1/G3) — รองรับทั้ง 2 พฤติกรรมของ Omise:
  // 1) มี signature + OMISE_WEBHOOK_SECRET -> ตรวจ HMAC (fail-closed)
  // 2) ไม่มี signature (พฤติกรรมจริงของ Omise) -> ยืนยัน event id ย้อนกลับที่ Omise API
  //    (วิธีที่ Omise แนะนำ) — ต้องมี OMISE_SECRET_KEY
  const signature = req.headers.get("x-omise-signature") || req.headers.get("X-Omise-Signature");
  const webhookSecret = process.env.OMISE_WEBHOOK_SECRET;

  let body: any;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  if (signature && webhookSecret) {
    const expected = crypto.createHmac("sha256", webhookSecret).update(rawBody).digest("hex");
    const received = Buffer.from(signature, "hex");
    const expectedBuf = Buffer.from(expected, "hex");
    if (received.length !== expectedBuf.length || !crypto.timingSafeEqual(received, expectedBuf)) {
      return NextResponse.json({ error: "Invalid signature" }, { status: 403 });
    }
  } else if (!process.env.OMISE_SECRET_KEY) {
    // ไม่มีวิธียืนยันเลย -> ปฏิเสธทุก request (อย่าเปิดรับ webhook ที่ยืนยันไม่ได้)
    return NextResponse.json(
      { error: "Webhook verification not configured (set OMISE_WEBHOOK_SECRET or OMISE_SECRET_KEY)" },
      { status: 503 }
    );
  } else {
    // ไม่มี signature (หรือมีแต่ตั้ง secret ไม่ครบ) -> ตรวจว่า event id เป็นของจริงโดย fetch กลับที่ Omise API
    const eventId = body?.data?.id;
    if (typeof eventId !== "string" || !eventId.startsWith("evnt_")) {
      return NextResponse.json({ error: "Invalid event id" }, { status: 403 });
    }
    try {
      const event = await omise.events.retrieve(eventId);
      if (!event || event.id !== eventId) {
        return NextResponse.json({ error: "Event not found" }, { status: 403 });
      }
    } catch {
      return NextResponse.json({ error: "Event verification failed" }, { status: 403 });
    }
  }

  try {
    const { key, data } = body;

    // C16: จัดการ event แต่ละชนิดผ่าน handler map (แทน if-chain)
    const handlers: Record<string, (data: any) => Promise<void>> = {
      "charge.complete": async (eventData) => {
        const userId = eventData?.metadata?.userId;
        const tier = eventData?.metadata?.tier || "BASIC";
        const amount = eventData?.amount || 0;

        if (userId) {
          // บันทึก PaymentRecord ด้วยสถานะจริงเท่านั้น (ห้ามบันทึก "successful" ก่อนเช็ค status)
          if (eventData.status === "successful") {
            await savePaymentRecord(userId, eventData.id, amount, "successful");
            await activateSubscription(userId, tier);
          } else if (typeof eventData.status === "string" && eventData.status.length > 0) {
            await savePaymentRecord(userId, eventData.id, amount, eventData.status);
          }
        }
      },
      "charge.failed": async (eventData) => {
        const userId = eventData?.metadata?.userId;
        if (userId) {
          await savePaymentRecord(userId, eventData.id, eventData.amount || 0, "failed");
        }
      },
      "charge.expired": async (eventData) => {
        const userId = eventData?.metadata?.userId;
        if (userId) {
          await savePaymentRecord(userId, eventData.id, eventData.amount || 0, "expired");
        }
      },
      "schedule.process": async (eventData) => {
        const userId = eventData?.metadata?.userId;
        const tier = eventData?.metadata?.tier || "BASIC";
        const chargeId = eventData?.charge;
        const amount = eventData?.amount || 0;

        if (userId) {
          if (eventData.status === "successful") {
            if (chargeId) await savePaymentRecord(userId, chargeId, amount, "successful");
            await activateSubscription(userId, tier);
          } else if (eventData.status === "failed") {
            if (chargeId) await savePaymentRecord(userId, chargeId, amount, "failed");
            await markSubscriptionPastDue(userId);
          }
        }
      },
      "schedule.expired": async (eventData) => {
        const userId = eventData?.metadata?.userId;
        if (userId) {
          await markSubscriptionExpired(userId);
        }
      },
    };

    // event ที่ไม่รู้จัก -> ไม่แตะ DB (ตอบ received ตามปกติ)
    const handler = handlers[key];
    if (handler) {
      await handler(data);
    }

    return NextResponse.json({ received: true });
  } catch (error: any) {
    // M17: log รายละเอียดฝั่ง server เท่านั้น — อย่าส่ง error.message กลับไป client
    console.error("Webhook processing error:", error);
    return NextResponse.json({ error: "Webhook error" }, { status: 500 });
  }
}
