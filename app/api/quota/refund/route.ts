import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// คืนโควตาที่หักไปแล้วเมื่องานประมวลผลล้มเหลว/ถูกยกเลิก (F11)
// เรียกจากฝั่ง Frontend หลัง backend ตอบ error ระหว่าง processing
export async function POST() {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;
  if (!session.user.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      usageQuotas: { orderBy: { periodStart: "desc" }, take: 1 },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const quota = user.usageQuotas[0];
  if (!quota) {
    // ไม่มีรอบโควตาให้คืน ถือว่าสำเร็จ (idempotent)
    return NextResponse.json({ success: true });
  }

  // ลดเฉพาะเมื่อ usedCount > 0 เพื่อกันค่าติดลบจากการ refund ซ้ำ
  await prisma.usageQuota.updateMany({
    where: { id: quota.id, usedCount: { gt: 0 } },
    data: { usedCount: { decrement: 1 } },
  });

  return NextResponse.json({ success: true });
}
