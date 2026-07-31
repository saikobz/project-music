import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET(req?: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  // L9: pagination — จำกัด 50 รายการต่อหน้า (cursor-based จาก id)
  const cursor = req ? new URL(req.url).searchParams.get("cursor") : null;

  const records = await prisma.paymentRecord.findMany({
    where: { userId: session.user.id },
    orderBy: { paidAt: "desc" },
    take: 50,
    ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
  });

  return NextResponse.json({
    payments: records.map((r) => ({
      id: r.id,
      amount: r.amount,
      currency: r.currency,
      status: r.status,
      paidAt: r.paidAt,
    })),
  });
}
