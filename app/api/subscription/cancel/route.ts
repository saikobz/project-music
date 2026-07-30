import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { omise } from "@/lib/omise";

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let password: string | undefined;
  try {
    const body = await req.json();
    password = body.password;
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

  if (user.password) {
    if (!password) {
      return NextResponse.json({ error: "Password is required" }, { status: 400 });
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return NextResponse.json({ error: "Incorrect password" }, { status: 401 });
    }
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
