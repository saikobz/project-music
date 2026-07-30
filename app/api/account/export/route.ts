import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      subscription: true,
      usageQuotas: { orderBy: { periodStart: "desc" } },
      projectRecords: { orderBy: { createdAt: "desc" } },
      accounts: { select: { provider: true, providerAccountId: true } },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const exportData = {
    exportedAt: new Date().toISOString(),
    profile: {
      name: user.name,
      email: user.email,
      createdAt: user.createdAt,
    },
    preferences: {
      theme: user.theme,
      language: user.language,
      emailNotifications: user.emailNotifications,
    },
    subscription: user.subscription
      ? {
          tier: user.subscription.tier,
          status: user.subscription.status,
          paymentMethod: user.subscription.paymentMethod,
          currentPeriodStart: user.subscription.currentPeriodStart,
          currentPeriodEnd: user.subscription.currentPeriodEnd,
        }
      : null,
    usageQuotas: user.usageQuotas.map((q) => ({
      monthlyQuota: q.monthlyQuota,
      usedCount: q.usedCount,
      periodStart: q.periodStart,
      periodEnd: q.periodEnd,
    })),
    projectHistory: user.projectRecords.map((r) => ({
      action: r.action,
      originalFilename: r.originalFilename,
      fileId: r.fileId,
      stems: r.stems,
      createdAt: r.createdAt,
    })),
    connectedAccounts: user.accounts.map((a) => ({
      provider: a.provider,
      providerAccountId: a.providerAccountId,
    })),
  };

  const dateStr = new Date().toISOString().split("T")[0];
  return new NextResponse(JSON.stringify(exportData, null, 2), {
    headers: {
      "Content-Type": "application/json",
      "Content-Disposition": `attachment; filename="harmoniq-export-${dateStr}.json"`,
    },
  });
}
