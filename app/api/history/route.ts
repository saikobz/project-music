import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

// TTL ของไฟล์ output ที่ backend จะลบอัตโนมัติ (จาก .env, default 1200 วิ = 20 นาที)
const TTL_SECONDS = parseInt(process.env.SEPARATE_TTL_SECONDS || "1200", 10);

export async function GET(req?: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  // L9: pagination — จำกัด 50 รายการต่อหน้า (cursor-based จาก id)
  const cursor = req ? new URL(req.url).searchParams.get("cursor") : null;

  const records = await prisma.projectRecord.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 50,
    ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
  });

  // record เก่าที่ยังไม่มี expiresAt -> คำนวณจาก createdAt + TTL เพื่อให้สถานะไฟล์ถูกต้อง
  const enriched = records.map((r) => ({
    ...r,
    expiresAt:
      r.expiresAt ??
      (r.fileId ? new Date(r.createdAt.getTime() + TTL_SECONDS * 1000) : null),
  }));

  return NextResponse.json({ records: enriched });
}

export async function POST(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  // L11: กัน body ที่ไม่ใช่ JSON -> 400 (เดิม req.json() throw -> 500)
  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { action, originalFilename, fileId, stems } = body || {};

  if (typeof action !== "string" || typeof originalFilename !== "string" || !action || !originalFilename) {
    return NextResponse.json(
      { error: "Missing required fields: action, originalFilename" },
      { status: 400 }
    );
  }

  // จำกัดขนาดชื่อไฟล์ และ validate stems ให้เป็น array of string
  if (originalFilename.length > 255) {
    return NextResponse.json({ error: "originalFilename is too long" }, { status: 400 });
  }
  const validStems = Array.isArray(stems)
    ? stems.filter((s): s is string => typeof s === "string").slice(0, 8)
    : null;
  const fileIdValue = typeof fileId === "string" ? fileId : null;

  const record = await prisma.projectRecord.create({
    data: {
      userId: session.user.id,
      action,
      originalFilename,
      fileId: fileIdValue,
      stems: validStems && validStems.length > 0 ? JSON.stringify(validStems) : null,
      // บันทึกเวลาหมดอายุเฉพาะ action ที่มีไฟล์ output (เช่น separate) ตาม TTL backend
      expiresAt: fileIdValue ? new Date(Date.now() + TTL_SECONDS * 1000) : null,
    },
  });

  return NextResponse.json({ status: "success", record });
}
