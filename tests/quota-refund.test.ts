/**
 * regression tests สำหรับ TU-7 (F11): /api/quota/refund
 * - คืนโควตาเฉพาะเมื่อ usedCount > 0 (กันค่าติดลบจากการ refund ซ้ำ)
 * - ไม่มีรอบโควตา -> สำเร็จแบบ idempotent
 *
 * @jest-environment node
 */
jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("next-auth", () => ({
  __esModule: true,
  getServerSession: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: { findUnique: jest.fn() },
    usageQuota: { updateMany: jest.fn() },
  },
}));

import { getServerSession } from "next-auth";
import { prisma } from "@/lib/prisma";
import { POST } from "../app/api/quota/refund/route";

const prismaMock = prisma as unknown as {
  user: { findUnique: jest.Mock };
  usageQuota: { updateMany: jest.Mock };
};

describe("Quota Refund (TU-7 F11)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getServerSession as jest.Mock).mockResolvedValue({ user: { id: "user-1" } });
    prismaMock.usageQuota.updateMany.mockResolvedValue({ count: 1 });
  });

  it("returns 401 when unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);

    const res = await POST();
    expect(res.status).toBe(401);
  });

  it("decrements usedCount for the latest quota period", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-1",
      usageQuotas: [{ id: "quota-1", usedCount: 3 }],
    });

    const res = await POST();
    expect(res.status).toBe(200);
    expect(prismaMock.usageQuota.updateMany).toHaveBeenCalledWith({
      where: { id: "quota-1", usedCount: { gt: 0 } },
      data: { usedCount: { decrement: 1 } },
    });
  });

  it("returns success without touching DB when no quota period exists", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce({ id: "user-1", usageQuotas: [] });

    const res = await POST();
    expect(res.status).toBe(200);
    expect(prismaMock.usageQuota.updateMany).not.toHaveBeenCalled();
  });

  it("does not decrement below zero (guarded by usedCount gt 0)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-1",
      usageQuotas: [{ id: "quota-1", usedCount: 0 }],
    });

    // updateMany with gt:0 จะไม่ match แถวที่ usedCount=0 -> count 0
    prismaMock.usageQuota.updateMany.mockResolvedValueOnce({ count: 0 });

    const res = await POST();
    expect(res.status).toBe(200);
  });
});
