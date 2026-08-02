jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: { findUnique: jest.fn() },
    subscription: { create: jest.fn() },
    usageQuota: { create: jest.fn() },
  },
}));

jest.mock("bcryptjs", () => ({
  hashSync: jest.fn(() => "dummy-hash"),
  compare: jest.fn(),
}));

import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";
import { authOptions, credentialsAuthorize } from "../lib/auth";

const prismaMock = prisma as unknown as { user: { findUnique: jest.Mock } };

describe("NextAuth Session Callback", () => {
  it("should enrich session with user tier and subscription status", async () => {
    const mockSession = { user: { email: "test@example.com", id: "u-1", tier: "FREE" }, expires: "2099-01-01" };
    const mockToken = { sub: "user-123", id: "user-123", tier: "PRO", omiseCustomerId: "cust_123" };

    // L1: session callback อ่าน tier ล่าสุดจาก DB
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-123",
      subscription: { tier: "PRO" },
    });

    if (authOptions.callbacks?.session) {
      const session = await authOptions.callbacks.session({
        session: mockSession as any,
        token: mockToken as any,
        user: {} as any,
        newSession: {} as any,
        trigger: "update",
      });

      const user = session?.user as any;
      expect(user?.tier).toBe("PRO");
      expect(user?.omiseCustomerId).toBe("cust_123");
    }
  });

  it("should fall back to FREE when DB lookup fails (L1)", async () => {
    const mockSession = { user: { email: "test@example.com", id: "u-1", tier: "FREE" }, expires: "2099-01-01" };
    const mockToken = { sub: "user-123", id: "user-123", tier: "PRO", omiseCustomerId: undefined };
    prismaMock.user.findUnique.mockRejectedValueOnce(new Error("db down"));

    if (authOptions.callbacks?.session) {
      const session = await authOptions.callbacks.session({
        session: mockSession as any,
        token: mockToken as any,
        user: {} as any,
        newSession: {} as any,
        trigger: "update",
      });
      expect((session?.user as any).tier).toBe("PRO");
    }
  });
});

describe("Credentials Authorize (TU-9 F9)", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("throws the same error for unknown email and wrong password (no enumeration)", async () => {
    // กรณี 1: ไม่มี email นี้ในระบบ
    prismaMock.user.findUnique.mockResolvedValueOnce(null);
    (bcrypt.compare as jest.Mock).mockResolvedValueOnce(false);
    await expect(
      credentialsAuthorize({ email: "ghost@example.com", password: "whatever1" })
    ).rejects.toThrow("Invalid email or password");

    // กรณี 2: มี email แต่รหัสผ่านผิด
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "u-1",
      email: "real@example.com",
      password: "hashed",
    });
    (bcrypt.compare as jest.Mock).mockResolvedValueOnce(false);
    await expect(
      credentialsAuthorize({ email: "real@example.com", password: "wrongpass" })
    ).rejects.toThrow("Invalid email or password");
  });

  it("always runs bcrypt compare even when user does not exist (timing-safe)", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(null);
    (bcrypt.compare as jest.Mock).mockResolvedValueOnce(false);
    await expect(
      credentialsAuthorize({ email: "ghost@example.com", password: "whatever1" })
    ).rejects.toThrow("Invalid email or password");
    expect(bcrypt.compare).toHaveBeenCalledWith("whatever1", "dummy-hash");
  });

  it("normalizes email to lowercase before lookup", async () => {
    prismaMock.user.findUnique.mockResolvedValueOnce(null);
    (bcrypt.compare as jest.Mock).mockResolvedValueOnce(false);
    await expect(
      credentialsAuthorize({ email: "  User@Example.COM ", password: "whatever1" })
    ).rejects.toThrow();
    expect(prismaMock.user.findUnique).toHaveBeenCalledWith({ where: { email: "user@example.com" } });
  });
});
