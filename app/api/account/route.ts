import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import bcrypt from "bcryptjs";
import { omise } from "@/lib/omise";

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

export async function DELETE(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let password: string | undefined;
  try {
    const body = await req.json();
    password = body.password;
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

  if (user.password) {
    if (!password) {
      return NextResponse.json({ error: "Password is required to delete account" }, { status: 400 });
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return NextResponse.json({ error: "Incorrect password" }, { status: 401 });
    }
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
