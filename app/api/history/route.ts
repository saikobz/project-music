import { NextResponse } from "next/server";
import { getServerSession } from "next-auth";
import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const records = await prisma.projectRecord.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
  });

  return NextResponse.json({ records });
}

export async function POST(req: Request) {
  const session = await getServerSession(authOptions);
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { action, originalFilename, fileId, stems } = body;

  if (!action || !originalFilename) {
    return NextResponse.json(
      { error: "Missing required fields: action, originalFilename" },
      { status: 400 }
    );
  }

  const record = await prisma.projectRecord.create({
    data: {
      userId: session.user.id,
      action,
      originalFilename,
      fileId: fileId || null,
      stems: stems ? JSON.stringify(stems) : null,
    },
  });

  return NextResponse.json({ status: "success", record });
}
