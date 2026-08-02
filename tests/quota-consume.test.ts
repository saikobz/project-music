/**
 * regression tests สำหรับ TU-5 (F2):
 * - ผู้ใช้ที่ยกเลิก/หมดอายุ/ครบรอบ ต้องไม่ได้สิทธิ์ tier แล้ว (เท่ากับ FREE)
 * - ผู้ใช้ ACTIVE + ยังไม่หมดรอบ ได้สิทธิ์ตาม tier จริง
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
    usageQuota: { upsert: jest.fn(), updateMany: jest.fn() },
  },
}));

import { getServerSession } from "next-auth";
import { prisma } from "@/lib/prisma";
import { POST } from "../app/api/quota/consume/route";

const prismaMock = prisma as unknown as {
  user: { findUnique: jest.Mock };
  usageQuota: { upsert: jest.Mock; updateMany: jest.Mock };
};

function futureDate(days: number): Date {
  return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
}

function makeUser(
  sub: { tier: string; status: string; currentPeriodEnd: Date | null } | null,
  usedCount = 0
) {
  return {
    id: "user-1",
    subscription: sub,
    usageQuotas: [
      {
        id: "quota-1",
        monthlyQuota: 15,
        usedCount,
        periodStart: futureDate(-5),
        periodEnd: futureDate(25),
      },
    ],
  };
}

describe("Quota Consume (TU-5 F2)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getServerSession as jest.Mock).mockResolvedValue({ user: { id: "user-1" } });
    // default: หักโควตาสำเร็จ (atomic update ชนะ)
    prismaMock.usageQuota.updateMany.mockResolvedValue({ count: 1 });
  });

  it("returns 401 when unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);

    const res = await POST();
    expect(res.status).toBe(401);
  });

  it("grants PRO unlimited quota when ACTIVE and within period", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "PRO", status: "ACTIVE", currentPeriodEnd: futureDate(10) })
    );

    const res = await POST();
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.quota.monthlyQuota).toBe(-1);
  });

  it("treats CANCELED PRO as FREE (F2)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "PRO", status: "CANCELED", currentPeriodEnd: futureDate(10) }, 1)
    );

    const res = await POST();
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.quota.monthlyQuota).toBe(3);
    expect(data.quota.usedCount).toBe(2);
  });

  it("treats EXPIRED BASIC as FREE (F2)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "BASIC", status: "EXPIRED", currentPeriodEnd: futureDate(-3) }, 2)
    );

    const res = await POST();
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.quota.monthlyQuota).toBe(3);
  });

  it("treats ACTIVE PRO with expired period as FREE (F2)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "PRO", status: "ACTIVE", currentPeriodEnd: futureDate(-1) }, 2)
    );

    const res = await POST();
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.quota.monthlyQuota).toBe(3);
  });

  it("blocks when FREE quota is exhausted", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "FREE", status: "ACTIVE", currentPeriodEnd: null }, 3)
    );
    prismaMock.usageQuota.updateMany.mockResolvedValueOnce({ count: 0 });

    const res = await POST();
    expect(res.status).toBe(403);
    // ตรวจว่าใช้ conditional update แบบ atomic (ห้าม update เฉยๆ)
    expect(prismaMock.usageQuota.updateMany).toHaveBeenCalledWith(
      expect.objectContaining({
        data: { usedCount: { increment: 1 } },
      })
    );
  });

  it("blocks CANCELED PRO user whose FREE quota is exhausted (F2)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(
      makeUser({ tier: "PRO", status: "CANCELED", currentPeriodEnd: futureDate(10) }, 3)
    );
    prismaMock.usageQuota.updateMany.mockResolvedValueOnce({ count: 0 });

    const res = await POST();
    expect(res.status).toBe(403);
  });

  it("succeeds for exactly one of two concurrent requests when only 1 slot left (F4)", async () => {
    // จำลอง TOCTOU: request 2 ตัวอ่าน usedCount=2 (เหลือ 1) พร้อมกัน
    // ตัวแรก updateMany ชนะ (count=1) ตัวที่สองแพ้ (count=0)
    prismaMock.user.findUnique.mockResolvedValue(
      makeUser({ tier: "FREE", status: "ACTIVE", currentPeriodEnd: null }, 2)
    );
    prismaMock.usageQuota.updateMany
      .mockResolvedValueOnce({ count: 1 })
      .mockResolvedValueOnce({ count: 0 });

    const [res1, res2] = await Promise.all([POST(), POST()]);
    expect(res1.status).toBe(200);
    expect(res2.status).toBe(403);
  });
});
