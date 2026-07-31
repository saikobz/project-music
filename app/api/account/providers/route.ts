import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";
import { OAUTH_PROVIDERS } from "@/lib/config";

export async function GET() {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const configuredProviders = OAUTH_PROVIDERS.map((p) => ({ ...p }));

  const accounts = await prisma.account.findMany({
    where: { userId: session.user.id },
  });
  const linkedProviders = new Set(accounts.map((a) => a.provider));

  const user = await prisma.user.findUnique({ where: { id: session.user.id } });
  const hasPassword = !!user?.password;

  const providers = [
    ...configuredProviders.map((p) => ({
      ...p,
      linked: linkedProviders.has(p.id),
    })),
    {
      id: "credentials",
      name: "Email & Password",
      icon: "mail",
      linked: hasPassword,
    },
  ];

  return NextResponse.json({ providers });
}

export async function DELETE(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const { provider } = await req.json();
  if (!provider) {
    return NextResponse.json({ error: "Provider is required" }, { status: 400 });
  }

  const accounts = await prisma.account.findMany({
    where: { userId: session.user.id },
  });
  const account = accounts.find((a) => a.provider === provider);
  if (!account) {
    return NextResponse.json({ error: "Account not found" }, { status: 404 });
  }

  // M20: กัน lockout ถาวร — ห้ามลบ provider ตัวสุดท้ายที่เหลืออยู่ถ้าไม่มีรหัสผ่านสำรอง
  const user = await prisma.user.findUnique({ where: { id: session.user.id } });
  const hasPassword = !!user?.password;
  if (accounts.length === 1 && !hasPassword) {
    return NextResponse.json(
      {
        error: "ไม่สามารถยกเลิกการเชื่อมต่อ Provider สุดท้ายได้ เนื่องจากบัญชีนี้ไม่มีรหัสผ่านสำรอง กรุณาตั้งรหัสผ่านก่อน",
      },
      { status: 400 }
    );
  }

  await prisma.account.delete({ where: { id: account.id } });

  return NextResponse.json({ success: true });
}
