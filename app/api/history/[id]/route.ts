import { NextResponse } from "next/server";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const { id } = await params;

  // L6: ลบแบบ atomic (deleteMany กับเงื่อนไข ownership พร้อมกัน)
  // กัน race: ถ้าถูก request อื่นลบแทรกกลางระหว่าง findUnique กับ delete
  const result = await prisma.projectRecord.deleteMany({
    where: { id, userId: session.user.id },
  });
  if (result.count === 0) {
    // เช็คว่า record ยังมีอยู่ไหมเพื่อแยก 404 (ไม่มี) กับ 403 (เป็นของคนอื่น)
    const record = await prisma.projectRecord.findUnique({ where: { id } });
    if (!record) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  return NextResponse.json({ status: "success" });
}
