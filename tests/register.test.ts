/**
 * regression tests สำหรับ TU-9 (M16): Register validation
 * - password สั้นกว่า 6 -> 400 (สอดคล้องกับ password route)
 * - email format ผิด -> 400
 * - name ไม่ใช่ string -> 400 (เดิม 500)
 * - email ซ้ำ -> 409 (เดิม 400/500)
 * - email ถูก normalize (lowercase/trim) ก่อนเช็คซ้ำและสร้าง
 *
 * @jest-environment node
 */
jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: { findUnique: jest.fn(), create: jest.fn() },
  },
}));

jest.mock("bcryptjs", () => ({
  hash: jest.fn().mockResolvedValue("hashed"),
}));

import { prisma } from "@/lib/prisma";
import { POST } from "../app/api/auth/register/route";

const prismaMock = prisma as unknown as {
  user: { findUnique: jest.Mock; create: jest.Mock };
};

function post(body: unknown) {
  return POST(
    new Request("http://localhost/api/auth/register", {
      method: "POST",
      body: JSON.stringify(body),
    })
  );
}

describe("Register (TU-9 M16)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    prismaMock.user.findUnique.mockResolvedValue(null);
    prismaMock.user.create.mockResolvedValue({ id: "user-1" });
  });

  it("rejects password shorter than 6 characters with 400", async () => {
    const res = await post({ email: "test@example.com", password: "abc" });
    expect(res.status).toBe(400);
    expect(prismaMock.user.create).not.toHaveBeenCalled();
  });

  it("rejects invalid email format with 400", async () => {
    const res = await post({ email: "not-an-email", password: "secret1" });
    expect(res.status).toBe(400);
  });

  it("rejects non-string name with 400 instead of crashing", async () => {
    const res = await post({ email: "test@example.com", password: "secret1", name: { evil: true } });
    expect(res.status).toBe(400);
    expect(prismaMock.user.create).not.toHaveBeenCalled();
  });

  it("returns 409 when email already registered", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce({ id: "existing" });
    const res = await post({ email: "test@example.com", password: "secret1" });
    expect(res.status).toBe(409);
  });

  it("normalizes email to lowercase before duplicate check and creation", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce({ id: "existing" });
    // A@X.com ต้องเจอผู้ใช้ a@x.com ที่ลงทะเบียนไว้แล้ว (case-insensitive)
    const res = await post({ email: "  A@X.com  ", password: "secret1" });
    expect(res.status).toBe(409);
    expect(prismaMock.user.findUnique).toHaveBeenCalledWith({ where: { email: "a@x.com" } });
  });

  it("creates user with normalized email and returns success", async () => {
    const res = await post({ email: "  User@Example.COM  ", password: "secret1" });
    expect(res.status).toBe(200);
    expect(prismaMock.user.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ email: "user@example.com" }),
      })
    );
  });
});
