import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const records = await prisma.paymentRecord.findMany({
    where: { userId: session.user.id },
    orderBy: { paidAt: "desc" },
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
