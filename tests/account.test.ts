jest.mock("@auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: {
      findUnique: jest.fn(),
      update: jest.fn(),
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
});
