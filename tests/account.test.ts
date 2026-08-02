jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: {
      findUnique: jest.fn(),
      update: jest.fn(),
      delete: jest.fn(),
    },
    account: {
      findMany: jest.fn(),
      findFirst: jest.fn(),
      delete: jest.fn(),
    },
  },
}));

jest.mock("bcryptjs", () => ({
  compare: jest.fn(),
  hash: jest.fn(),
  hashSync: jest.fn(() => "dummy-hash"),
}));

jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

import { GET } from "../app/api/account/route";
import { PUT as PutProfile } from "../app/api/account/profile/route";
import { PUT as PutPassword } from "../app/api/account/password/route";
import { GET as GetProviders } from "../app/api/account/providers/route";
import { DELETE as DeleteProviders } from "../app/api/account/providers/route";
import { PUT as PutPreferences } from "../app/api/account/preferences/route";
import { getServerSession } from "next-auth";
import { DELETE } from "../app/api/account/route";
import { GET as GetExport } from "../app/api/account/export/route";
import { POST as PostCancel } from "../app/api/subscription/cancel/route";
import { GET as GetPaymentHistory } from "../app/api/subscription/history/route";
const { prisma } = require("@/lib/prisma");

describe("Account API Endpoint", () => {
  it("should return 401 if user is unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);

    const res = await GET();
    expect(res.status).toBe(401);
  });
});

describe("Account API — PUT /api/account/profile", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ name: "Test" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(401);
  });
});

describe("Account API — PUT /api/account/password", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({ currentPassword: "old", newPassword: "new123" }),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if passwords missing", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({}),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(400);
  });

  it("should return 400 if new password too short", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    const req = new Request("http://localhost/api/account/password", {
      method: "PUT",
      body: JSON.stringify({ currentPassword: "old", newPassword: "ab" }),
    });
    const res = await PutPassword(req);
    expect(res.status).toBe(400);
  });
});

describe("Account API — GET /api/account/providers", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GetProviders();
    expect(res.status).toBe(401);
  });
});

describe("Account API — DELETE /api/account/providers", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/providers", {
      method: "DELETE",
      body: JSON.stringify({ provider: "google" }),
    });
    const res = await DeleteProviders(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if provider missing", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    const req = new Request("http://localhost/api/account/providers", {
      method: "DELETE",
      body: JSON.stringify({}),
    });
    const res = await DeleteProviders(req);
    expect(res.status).toBe(400);
  });

  it("should block unlinking the last provider when no password backup (M20)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.account.findMany.mockResolvedValueOnce([{ id: "acc-1", provider: "google" }]);
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: null });

    const req = new Request("http://localhost/api/account/providers", {
      method: "DELETE",
      body: JSON.stringify({ provider: "google" }),
    });
    const res = await DeleteProviders(req);
    expect(res.status).toBe(400);
    expect(prisma.account.delete).not.toHaveBeenCalled();
  });

  it("should allow unlinking provider when password backup exists (M20)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.account.findMany.mockResolvedValueOnce([{ id: "acc-1", provider: "google" }]);
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: "hashed" });
    prisma.account.delete.mockResolvedValueOnce({});

    const req = new Request("http://localhost/api/account/providers", {
      method: "DELETE",
      body: JSON.stringify({ provider: "google" }),
    });
    const res = await DeleteProviders(req);
    expect(res.status).toBe(200);
    expect(prisma.account.delete).toHaveBeenCalledWith({ where: { id: "acc-1" } });
  });

  it("should allow unlinking one provider when another remains (M20)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.account.findMany.mockResolvedValueOnce([
      { id: "acc-1", provider: "google" },
      { id: "acc-2", provider: "line" },
    ]);
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: null });
    prisma.account.delete.mockResolvedValueOnce({});

    const req = new Request("http://localhost/api/account/providers", {
      method: "DELETE",
      body: JSON.stringify({ provider: "google" }),
    });
    const res = await DeleteProviders(req);
    expect(res.status).toBe(200);
  });
});

describe("Account API — PUT /api/account/preferences", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account/preferences", {
      method: "PUT",
      body: JSON.stringify({ theme: "DARK" }),
    });
    const res = await PutPreferences(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if invalid theme value", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    const req = new Request("http://localhost/api/account/preferences", {
      method: "PUT",
      body: JSON.stringify({ theme: "BLUE" }),
    });
    const res = await PutPreferences(req);
    expect(res.status).toBe(400);
  });
});

describe("Account API — Positive path tests", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("PUT /api/account/profile — should update name and return 200", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.user.update.mockResolvedValueOnce({ name: "New Name", email: "test@test.com", image: null });
    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ name: "New Name" }),
    });
    const res = await PutProfile(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.user.name).toBe("New Name");
  });

  it("should require currentPassword to change email (M21)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "old@test.com" } });
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: "hashed", email: "old@test.com" });

    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ email: "new@test.com" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(400);
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it("should reject wrong password when changing email (M21)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "old@test.com" } });
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: "hashed", email: "old@test.com" });
    (require("bcryptjs").compare as jest.Mock).mockResolvedValueOnce(false);

    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ email: "new@test.com", currentPassword: "wrong" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(401);
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it("should change email with correct password, normalized (M21)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "old@test.com" } });
    prisma.user.findUnique.mockResolvedValueOnce({ id: "user-1", password: "hashed", email: "old@test.com" });
    (require("bcryptjs").compare as jest.Mock).mockResolvedValueOnce(true);
    // findUnique ตัวที่ 2 = เช็ค email ซ้ำ -> null (ไม่มี)
    prisma.user.findUnique.mockResolvedValueOnce(null);
    prisma.user.update.mockResolvedValueOnce({ name: null, email: "new@test.com", image: null });

    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ email: "  NEW@Test.COM  ", currentPassword: "correct" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(200);
    expect(prisma.user.update).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ email: "new@test.com" }),
      })
    );
  });

  it("should return 409 when email is already in use (M21)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "old@test.com" } });
    prisma.user.findUnique
      .mockResolvedValueOnce({ id: "user-1", password: "hashed", email: "old@test.com" }) // user lookup
      .mockResolvedValueOnce({ id: "user-2", email: "new@test.com" }); // duplicate check
    (require("bcryptjs").compare as jest.Mock).mockResolvedValueOnce(true);

    const req = new Request("http://localhost/api/account/profile", {
      method: "PUT",
      body: JSON.stringify({ email: "new@test.com", currentPassword: "correct" }),
    });
    const res = await PutProfile(req);
    expect(res.status).toBe(409);
    expect(prisma.user.update).not.toHaveBeenCalled();
  });

  it("PUT /api/account/preferences — should save theme and return 200", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.user.update.mockResolvedValueOnce({
      theme: "LIGHT",
      language: "TH",
      emailNotifications: true,
    });
    const req = new Request("http://localhost/api/account/preferences", {
      method: "PUT",
      body: JSON.stringify({ theme: "LIGHT" }),
    });
    const res = await PutPreferences(req);
    const body = await res.json();
    expect(res.status).toBe(200);
    expect(body.preferences.theme).toBe("LIGHT");
  });
});

describe("Account API — DELETE /api/account", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/account", { method: "DELETE" });
    const res = await DELETE(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if password missing for credential user", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1", email: "test@test.com" } });
    prisma.user.findUnique = jest.fn().mockResolvedValue({ id: "user-1", password: "hashed", subscription: null });
    const req = new Request("http://localhost/api/account", {
      method: "DELETE",
      body: JSON.stringify({}),
    });
    const res = await DELETE(req);
    expect(res.status).toBe(400);
  });

  it("should require fresh reauth for OAuth-only user (M19+)", async () => {
    (getServerSession as jest.Mock).mockResolvedValue({ user: { id: "user-1", email: "oauth@test.com" } });
    prisma.user.findUnique.mockResolvedValue({ id: "user-1", password: null, email: "oauth@test.com", subscription: null });

    // ไม่มี reauthAt -> 403
    const res1 = await DELETE(new Request("http://localhost/api/account", { method: "DELETE", body: JSON.stringify({}) }));
    expect(res1.status).toBe(403);

    // reauthAt เก่าเกิน 5 นาที -> 403
    (getServerSession as jest.Mock).mockResolvedValue({
      user: { id: "user-1", email: "oauth@test.com", reauthAt: Date.now() - 6 * 60 * 1000 },
    });
    const res2 = await DELETE(new Request("http://localhost/api/account", { method: "DELETE", body: JSON.stringify({}) }));
    expect(res2.status).toBe(403);
    expect(prisma.user.delete).not.toHaveBeenCalled();
  });

  it("should delete account when OAuth-only user re-authenticated recently (M19+)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "oauth@test.com", reauthAt: Date.now() },
    });
    prisma.user.findUnique.mockResolvedValue({ id: "user-1", password: null, email: "oauth@test.com", subscription: null });
    prisma.user.delete.mockResolvedValue({});

    const req = new Request("http://localhost/api/account", {
      method: "DELETE",
      body: JSON.stringify({}),
    });
    const res = await DELETE(req);
    expect(res.status).toBe(200);
    expect(prisma.user.delete).toHaveBeenCalledWith({ where: { id: "user-1" } });
  });
});

describe("Account API — GET /api/account/export", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GetExport();
    expect(res.status).toBe(401);
  });
});

describe("Subscription API — POST /api/subscription/cancel", () => {
  beforeEach(() => { jest.clearAllMocks(); });

  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/subscription/cancel", {
      method: "POST",
      body: JSON.stringify({ password: "test123" }),
    });
    const res = await PostCancel(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if no active paid subscription", async () => {
    const { prisma: p } = require("@/lib/prisma");
    (getServerSession as jest.Mock).mockResolvedValueOnce({ user: { id: "user-1" } });
    p.user.findUnique.mockResolvedValueOnce({
      id: "user-1",
      password: "hashed",
      subscription: { tier: "FREE", status: "ACTIVE" },
    });
    const req = new Request("http://localhost/api/subscription/cancel", {
      method: "POST",
      body: JSON.stringify({ password: "test123" }),
    });
    const res = await PostCancel(req);
    expect(res.status).toBe(400);
  });

  it("should require fresh reauth for OAuth-only user cancelling (M19+)", async () => {
    const { prisma: p } = require("@/lib/prisma");
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "oauth@test.com" },
    });
    p.user.findUnique.mockResolvedValueOnce({
      id: "user-1",
      password: null,
      email: "oauth@test.com",
      subscription: { tier: "PRO", status: "ACTIVE" },
    });

    const req = new Request("http://localhost/api/subscription/cancel", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const res = await PostCancel(req);
    expect(res.status).toBe(403);
  });
});

describe("Subscription API — GET /api/subscription/history", () => {
  beforeEach(() => { jest.clearAllMocks(); });

  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GetPaymentHistory();
    expect(res.status).toBe(401);
  });
});
