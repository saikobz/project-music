import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { omise } from "@/lib/omise";
import { prisma } from "@/lib/prisma";
import { getTierPrice } from "@/lib/subscription";

// C16: แยก flow เก็บเงินรายเดือนผ่านบัตรเครดิตเป็นฟังก์ชันเดียว
// (F3 ทำลาย schedule เก่า, F7 attach บัตรใหม่, F6 status PENDING จนกว่า webhook ยืนยัน)
async function createCardSubscription(
  userId: string,
  email: string,
  tier: string,
  cardToken: string,
  amount: number
): Promise<{ scheduleId: string }> {
  const dbUser = await prisma.user.findUnique({
    where: { id: userId },
    include: { subscription: true },
  });

  // F3: ทำลาย schedule เก่าก่อนสร้างใหม่ เพื่อไม่ให้ลูกค้าถูก charge ซ้อนรายเดือน
  if (dbUser?.subscription?.omiseScheduleId) {
    try {
      await omise.schedules.destroy(dbUser.subscription.omiseScheduleId);
    } catch {
      // schedule เก่าอาจถูกลบไปแล้ว ปล่อยผ่าน
    }
  }

  let customerId = dbUser?.omiseCustomerId ?? null;
  if (!customerId) {
    const customer = await omise.customers.create({
      email: email || "",
      card: cardToken,
    });
    customerId = customer.id;
    await prisma.user.update({
      where: { id: userId },
      data: { omiseCustomerId: customerId },
    });
  } else {
    // F7: มี customer อยู่แล้ว + ผู้ใช้กรอกบัตรใหม่ -> attach การ์ดใหม่เข้ากับ customer เดิม
    // (เดิมเพิกเฉยการ์ดใหม่และหักบัตรใบเก่าเงียบๆ)
    await omise.customers.update(customerId, { card: cardToken });
  }

  const startDate = new Date();
  startDate.setDate(startDate.getDate() + 1);
  const endDate = new Date();
  endDate.setFullYear(endDate.getFullYear() + 1);

  const schedule = await omise.schedules.create({
    every: 1,
    period: "month",
    start_date: startDate.toISOString().split("T")[0],
    end_date: endDate.toISOString().split("T")[0],
    // metadata สำคัญมาก: webhook schedule.process ใช้ผูก charge รายเดือนกับ user
    charge: {
      customer: customerId,
      amount,
      metadata: { userId, tier },
    } as any,
  } as any);

  // F6: ยังไม่ให้สิทธิ์ ACTIVE จนกว่า webhook charge.complete successful จะมา
  // (เดิมเปิดสิทธิ์ทันทีทั้งที่ยังไม่เคยมี charge สำเร็จแม้แต่ครั้งเดียว)
  await prisma.subscription.upsert({
    where: { userId },
    update: { tier, status: "PENDING", paymentMethod: "CREDIT_CARD", omiseScheduleId: schedule.id },
    create: { userId, tier, status: "PENDING", paymentMethod: "CREDIT_CARD", omiseScheduleId: schedule.id },
  });

  return { scheduleId: schedule.id };
}

export async function POST(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const body = await req.json();
  const { tier, paymentMethod, cardToken } = body;

  // ตรวจสอบ tier ให้ตรงกับแพ็กเกจที่ขายจริง (กัน client ส่ง tier มั่ว -> ราคา/สิทธิ์ผิด)
  const amount = getTierPrice(tier);
  if (amount === null) {
    return NextResponse.json({ error: "Invalid tier" }, { status: 400 });
  }

  try {
    if (paymentMethod === "PROMPTPAY") {
      const charge = await omise.charges.create({
        amount,
        currency: "thb",
        source: { type: "promptpay" } as any,
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
      const { scheduleId } = await createCardSubscription(
        session.user.id,
        session.user.email || "",
        tier,
        cardToken,
        amount
      );
      return NextResponse.json({ success: true, scheduleId });
    }

    return NextResponse.json({ error: "Invalid payment method" }, { status: 400 });
  } catch (error: any) {
    // M17: log รายละเอียดฝั่ง server เท่านั้น — อย่าส่ง error.message (Omise detail) กลับไป client
    console.error("Checkout failed:", error);
    return NextResponse.json({ error: "Checkout failed" }, { status: 500 });
  }
}
