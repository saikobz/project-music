import { NextResponse } from "next/server";
import { requireSession, verifyAccountAuth } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { omise } from "@/lib/omise";

export async function POST(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  let password: string | undefined;
  let confirmEmail: string | undefined;
  try {
    const body = await req.json();
    password = body.password;
    confirmEmail = body.confirmEmail;
  } catch {}

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: { subscription: true },
  });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  if (!user.subscription || user.subscription.tier === "FREE") {
    return NextResponse.json({ error: "No active paid subscription" }, { status: 400 });
  }

  // C10 + M19+: ตรวจยืนยันตัวตนก่อนยกเลิก
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

  if (user.subscription.omiseScheduleId) {
    try {
      await omise.schedules.destroy(user.subscription.omiseScheduleId);
    } catch {
      // Schedule may already be destroyed — continue
    }
  }

  await prisma.subscription.update({
    where: { userId: session.user.id },
    data: { status: "CANCELED", omiseScheduleId: null },
  });

  return NextResponse.json({
    success: true,
    message: "Subscription cancelled. You remain on your current tier until the period ends.",
  });
}
