import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";
import { PERIOD_MS } from "@/lib/config";

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));

    // normalize email (lowercase + trim) เพื่อให้ unique constraint ทำงานแบบ case-insensitive
    const rawEmail = typeof body?.email === "string" ? body.email.trim().toLowerCase() : "";
    const rawPassword = typeof body?.password === "string" ? body.password : "";
    const rawName = typeof body?.name === "string" ? body.name.trim() : "";

    if (!rawEmail || !rawPassword) {
      return NextResponse.json({ error: "Email and password are required" }, { status: 400 });
    }

    if (body?.name !== undefined && typeof body.name !== "string") {
      return NextResponse.json({ error: "Name must be a string" }, { status: 400 });
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(rawEmail)) {
      return NextResponse.json({ error: "รูปแบบอีเมลไม่ถูกต้อง" }, { status: 400 });
    }

    // ให้สอดคล้องกับ /api/account/password ที่บังคับขั้นต่ำ 6 ตัว
    if (rawPassword.length < 6) {
      return NextResponse.json(
        { error: "รหัสผ่านต้องมีความยาวอย่างน้อย 6 ตัวอักษร" },
        { status: 400 }
      );
    }

    const existingUser = await prisma.user.findUnique({ where: { email: rawEmail } });
    if (existingUser) {
      return NextResponse.json({ error: "Email already registered" }, { status: 409 });
    }

    const hashedPassword = await bcrypt.hash(rawPassword, 10);
    const user = await prisma.user.create({
      data: {
        name: rawName || rawEmail.split("@")[0],
        email: rawEmail,
        password: hashedPassword,
        subscription: { create: { tier: "FREE", status: "ACTIVE" } },
        usageQuotas: {
          create: {
            monthlyQuota: 3,
            usedCount: 0,
            periodStart: new Date(),
            periodEnd: new Date(Date.now() + PERIOD_MS),
          },
        },
      },
    });

    return NextResponse.json({ success: true, userId: user.id });
  } catch (error: any) {
    // race: มี request อื่นสร้าง email นี้แทรกเข้ามา
    if (error?.code === "P2002") {
      return NextResponse.json({ error: "Email already registered" }, { status: 409 });
    }
    return NextResponse.json({ error: "Registration failed" }, { status: 500 });
  }
}
