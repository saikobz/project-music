import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { PERIOD_MS } from "@/lib/config";
import { getEffectiveTier, getMonthlyQuota } from "@/lib/subscription";

// ตรวจสอบและนับโควตาการใช้งานของผู้ใช้ที่ Login แล้ว (จัดการผ่าน Database แทนการนับตาม IP ของ Guest)
export async function POST() {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;
  if (!session.user.id) {
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

  // F2: สิทธิ์ tier ต้องอิงสถานะจริง (ACTIVE + ยังไม่หมดรอบ) ไม่ใช่แค่ค่าใน DB
  // ผู้ใช้ที่ยกเลิก/หมดอายุ/ครบรอบแล้วต้องได้สิทธิ์เท่ากับ FREE
  const tier = getEffectiveTier(user.subscription);
  const monthlyQuota = getMonthlyQuota(tier);

  // ถ้ายังไม่มี UsageQuota (เช่น OAuth) หรือหมดรอบแล้ว ให้สร้างรอบใหม่ (รีเซ็ตรายเดือน)
  // ใช้ upsert บน unique (userId, periodStart) กัน race ที่ request พร้อมกันสร้าง period ซ้ำ
  const now = new Date();
  let quota = user.usageQuotas[0];
  if (!quota || (quota.periodEnd && new Date(quota.periodEnd) <= now)) {
    const periodStart = now;
    quota = await prisma.usageQuota.upsert({
      where: {
        userId_periodStart: { userId: user.id, periodStart },
      },
      update: {},
      create: {
        userId: user.id,
        monthlyQuota,
        usedCount: 0,
        periodStart,
        periodEnd: new Date(now.getTime() + PERIOD_MS),
      },
    });
  }

  // ใช้ค่า monthlyQuota ที่คำนวณจาก effective tier เสมอ
  // (ห้ามใช้ quota.monthlyQuota จาก record เก่า เพราะ stale เมื่อ tier เปลี่ยน เช่น BASIC->FREE, PRO->FREE)
  const max = monthlyQuota;

  // F4: หักโควตาแบบ atomic (conditional update) — กัน TOCTOU race
  // เมื่อมี request พร้อมกัน 2 ตัว อ่านค่าเดียวกัน ผ่านเช็คพร้อมกัน แล้วใช้เกินโควตา
  const result = await prisma.usageQuota.updateMany({
    where: {
      id: quota.id,
      OR: [
        { monthlyQuota: -1 }, // PRO ไม่จำกัด
        { usedCount: { lt: max } },
      ],
    },
    data: { usedCount: { increment: 1 } },
  });

  if (result.count === 0) {
    return NextResponse.json(
      {
        error: `โควตาประมวลผลฟรีสำหรับผู้ใช้ ${tier} เต็มแล้ว (${quota.usedCount}/${max} เพลง) กรุณาสมัครสมาชิกเพื่อใช้งานต่อ`,
        quota: { monthlyQuota: max, usedCount: quota.usedCount },
      },
      { status: 403 }
    );
  }

  return NextResponse.json({ success: true, quota: { monthlyQuota: max, usedCount: quota.usedCount + 1 } });
}
