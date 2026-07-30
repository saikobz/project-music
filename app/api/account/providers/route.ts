import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const configuredProviders = [
    { id: "google", name: "Google", icon: "google" },
    { id: "facebook", name: "Facebook", icon: "facebook" },
    { id: "line", name: "LINE", icon: "line" },
  ];

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
