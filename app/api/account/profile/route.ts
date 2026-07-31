import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { requireSession } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export async function PUT(req: Request) {
  const { session, response: authResponse } = await requireSession();
  if (authResponse) return authResponse;

  const body = await req.json().catch(() => ({}));
  const { name, email, image } = body;

  const changes: Record<string, unknown> = {};
  if (name !== undefined) changes.name = name;

  if (email !== undefined) {
    const newEmail = typeof email === "string" ? email.trim().toLowerCase() : "";
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(newEmail)) {
      return NextResponse.json({ error: "รูปแบบอีเมลไม่ถูกต้อง" }, { status: 400 });
    }

    // เปลี่ยนอีเมลเท่านั้น (ถ้าเป็น email เดิมก็ข้ามขั้นตอนยืนยัน)
    if (newEmail !== (session.user.email || "").toLowerCase()) {
      // M21: ต้องยืนยันตัวตนก่อนเปลี่ยน email
      // - ผู้ใช้ที่มี password: ต้องยืนยันด้วย currentPassword
      // - ผู้ใช้ OAuth-only: ต้องกรอกอีเมลเดิมของบัญชี (confirmEmail)
      const user = await prisma.user.findUnique({ where: { id: session.user.id } });
      if (!user) {
        return NextResponse.json({ error: "User not found" }, { status: 404 });
      }

      if (user.password) {
        if (typeof body.currentPassword !== "string" || !body.currentPassword) {
          return NextResponse.json(
            { error: "ต้องยืนยันรหัสผ่านเพื่อเปลี่ยนอีเมล" },
            { status: 400 }
          );
        }
        const isValid = await bcrypt.compare(body.currentPassword, user.password);
        if (!isValid) {
          return NextResponse.json({ error: "รหัสผ่านไม่ถูกต้อง" }, { status: 401 });
        }
      } else {
        const confirmed =
          typeof body.confirmEmail === "string" &&
          body.confirmEmail.trim().toLowerCase() === user.email.toLowerCase();
        if (!confirmed) {
          return NextResponse.json(
            { error: "กรุณากรอกอีเมลเดิมของบัญชีเพื่อยืนยัน" },
            { status: 400 }
          );
        }
      }

      // ตรวจ email ซ้ำ (เช็คก่อน + กัน race ด้วย P2002 ใน catch)
      const existing = await prisma.user.findUnique({ where: { email: newEmail } });
      if (existing) {
        return NextResponse.json({ error: "Email already in use" }, { status: 409 });
      }
      changes.email = newEmail;
    }
  }

  if (image !== undefined) changes.image = image;

  try {
    const user = await prisma.user.update({
      where: { id: session.user.id },
      data: changes,
    });
    return NextResponse.json({
      user: { name: user.name, email: user.email, image: user.image },
    });
  } catch (error: any) {
    // race: มี request อื่นเปลี่ยน email ไปแล้ว -> unique constraint ชน
    if (error?.code === "P2002") {
      return NextResponse.json({ error: "Email already in use" }, { status: 409 });
    }
    return NextResponse.json({ error: "Failed to update profile" }, { status: 500 });
  }
}
