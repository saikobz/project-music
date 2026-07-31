import { NextResponse } from "next/server";
import { requireSession, verifyAccountAuth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { omise } from "@/lib/omise";
import { getEffectiveTier, getMonthlyQuota } from "@/lib/subscription";

export async function GET() {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

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

  // ใช้ effective tier (อิงสถานะ ACTIVE + รอบไม่หมด) เพื่อไม่ให้แสดง quota ผิดหลังยกเลิก/หมดอายุ
  const effectiveTier = getEffectiveTier(user.subscription);
  const currentQuota = user.usageQuotas[0] || {
    monthlyQuota: getMonthlyQuota(effectiveTier),
    usedCount: 0,
  };

  // L4: ไม่ส่ง omiseScheduleId (internal ID) กลับไปที่ client
  const { omiseScheduleId: _omit, ...publicSubscription } = user.subscription || {
    tier: "FREE",
    status: "ACTIVE",
  };

  return NextResponse.json({
    user: {
      id: user.id,
      name: user.name,
      email: user.email,
      image: user.image,
      createdAt: user.createdAt,
      hasPassword: !!user.password,
    },
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
    subscription: publicSubscription,
    quota: currentQuota,
  });
}

export async function DELETE(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  let password: string | undefined;
  let confirmEmail: string | undefined;
  try {
    const body = await req.json();
    password = body.password;
    confirmEmail = body.confirmEmail;
  } catch {
    // Body may be empty or malformed — proceed
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: { subscription: true },
  });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  // C10 + M19+: ตรวจยืนยันตัวตนก่อนลบ
  // (password สำหรับผู้ใช้ที่มี / re-auth ล่าสุดสำหรับ OAuth-only)
  const authError = await verifyAccountAuth(
    user,
    password,
    confirmEmail,
    (session.user as { reauthAt?: number }).reauthAt
  );
  if (authError) {
    return NextResponse.json({ error: authError.error }, { status: authError.status });
  }

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
