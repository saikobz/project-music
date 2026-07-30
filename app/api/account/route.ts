import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

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

  const currentQuota = user.usageQuotas[0] || {
    monthlyQuota: user.subscription?.tier === "PRO" ? -1 : user.subscription?.tier === "BASIC" ? 15 : 3,
    usedCount: 0,
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
    subscription: user.subscription || { tier: "FREE", status: "ACTIVE" },
    quota: currentQuota,
  });
}
