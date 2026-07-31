import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// ตรวจสอบและนับโควตาการใช้งานของผู้ใช้ที่ Login แล้ว (จัดการผ่าน Database แทนการนับตาม IP ของ Guest)
export async function POST() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      subscription: true,
      usageQuotas: { orderBy: { periodStart: "desc" }, take: 1 },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const tier = user.subscription?.tier || "FREE";
  const monthlyQuota = tier === "PRO" ? -1 : tier === "BASIC" ? 15 : 3;

  // ถ้ายังไม่มี UsageQuota (เช่น OAuth) หรือหมดรอบแล้ว ให้สร้างรอบใหม่ (รีเซ็ตรายเดือน)
  const now = new Date();
  let quota = user.usageQuotas[0];
  if (!quota || (quota.periodEnd && new Date(quota.periodEnd) <= now)) {
    quota = await prisma.usageQuota.create({
      data: {
        userId: user.id,
        monthlyQuota,
        usedCount: 0,
        periodStart: now,
        periodEnd: new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000),
      },
    });
  }

  const max = quota.monthlyQuota;
  if (max !== -1 && quota.usedCount >= max) {
    return NextResponse.json(
      {
        error: `โควตาประมวลผลฟรีสำหรับผู้ใช้ ${tier} เต็มแล้ว (${quota.usedCount}/${max} เพลง) กรุณาสมัครสมาชิกเพื่อใช้งานต่อ`,
        quota: { monthlyQuota: max, usedCount: quota.usedCount },
      },
      { status: 403 }
    );
  }

  const nextCount = quota.usedCount + 1;
  await prisma.usageQuota.update({
    where: { id: quota.id },
    data: { usedCount: nextCount },
  });

  return NextResponse.json({ success: true, quota: { monthlyQuota: max, usedCount: nextCount } });
}
