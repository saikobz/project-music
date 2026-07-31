jest.mock("@auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    projectRecord: {
      findMany: jest.fn(),
      create: jest.fn(),
      deleteMany: jest.fn(),
      findUnique: jest.fn(),
    },
  },
}));

import { GET, POST } from "../app/api/history/route";
import { DELETE } from "../app/api/history/[id]/route";
import { getServerSession } from "next-auth";
import { prisma } from "@/lib/prisma";

const prismaMock = prisma as unknown as {
  projectRecord: {
    findMany: jest.Mock;
    create: jest.Mock;
    deleteMany: jest.Mock;
    findUnique: jest.Mock;
  };
};

const authedSession = { user: { id: "user-1", email: "test@test.com" } };

describe("History API — GET", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GET();
    expect(res.status).toBe(401);
  });

  it("should enrich legacy records with expiresAt = createdAt + TTL when fileId exists", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(authedSession);
    const createdAt = new Date("2026-07-31T10:00:00Z");
    prismaMock.projectRecord.findMany.mockResolvedValueOnce([
      { id: "rec-1", fileId: "abc123", createdAt, expiresAt: null },
      { id: "rec-2", fileId: null, createdAt, expiresAt: null },
    ]);

    const res = await GET(new Request("http://localhost/api/history"));
    const body = await res.json();

    expect(body.records[0].expiresAt).toBe(new Date(createdAt.getTime() + 1200 * 1000).toISOString());
    expect(body.records[1].expiresAt).toBeNull();
  });
});

describe("History API — POST", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getServerSession as jest.Mock).mockResolvedValue(authedSession);
    prismaMock.projectRecord.create.mockResolvedValue({ id: "rec-1" });
  });

  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({ action: "separate", originalFilename: "test.wav" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(401);
  });

  it("should return 400 if missing required fields", async () => {
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
  });

  it("should return 400 for non-JSON body instead of 500 (L11)", async () => {
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: "this is not json",
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
    expect(prismaMock.projectRecord.create).not.toHaveBeenCalled();
  });

  it("should filter non-string stems (L11)", async () => {
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({
        action: "separate",
        originalFilename: "test.wav",
        stems: ["vocals", 123, null, "drums"],
      }),
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.projectRecord.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ stems: JSON.stringify(["vocals", "drums"]) }),
      })
    );
  });

  it("should store stems as null when not an array (L11)", async () => {
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({ action: "separate", originalFilename: "test.wav", stems: "vocals" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.projectRecord.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ stems: null }),
      })
    );
  });

  it("should set expiresAt = now + TTL when fileId is provided", async () => {
    const before = Date.now();
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({
        action: "separate",
        originalFilename: "test.wav",
        fileId: "file-1",
      }),
    });
    const res = await POST(req);
    const after = Date.now();
    expect(res.status).toBe(200);

    const data = prismaMock.projectRecord.create.mock.calls[0][0].data;
    expect(data.fileId).toBe("file-1");
    const expiresAtMs = (data.expiresAt as Date).getTime();
    expect(expiresAtMs).toBeGreaterThanOrEqual(before + 1200 * 1000 - 5000);
    expect(expiresAtMs).toBeLessThanOrEqual(after + 1200 * 1000 + 5000);
  });

  it("should set expiresAt = null when no fileId is provided", async () => {
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({ action: "apply-eq-ai", originalFilename: "test.wav" }),
    });
    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.projectRecord.create).toHaveBeenCalledWith(
      expect.objectContaining({
        data: expect.objectContaining({ expiresAt: null }),
      })
    );
  });
});

describe("History API — DELETE /api/history/[id]", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (getServerSession as jest.Mock).mockResolvedValue(authedSession);
  });

  function deleteReq(id: string) {
    return DELETE(new Request(`http://localhost/api/history/${id}`, { method: "DELETE" }), {
      params: Promise.resolve({ id }),
    });
  }

  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await deleteReq("rec-1");
    expect(res.status).toBe(401);
  });

  it("should return 404 when record does not exist (L6)", async () => {
    prismaMock.projectRecord.deleteMany.mockResolvedValueOnce({ count: 0 });
    prismaMock.projectRecord.findUnique.mockResolvedValueOnce(null);

    const res = await deleteReq("ghost-id");
    expect(res.status).toBe(404);
  });

  it("should return 403 when record belongs to another user (L6)", async () => {
    prismaMock.projectRecord.deleteMany.mockResolvedValueOnce({ count: 0 });
    prismaMock.projectRecord.findUnique.mockResolvedValueOnce({
      id: "rec-1",
      userId: "other-user",
    });

    const res = await deleteReq("rec-1");
    expect(res.status).toBe(403);
  });

  it("should delete atomically with ownership condition (L6)", async () => {
    prismaMock.projectRecord.deleteMany.mockResolvedValueOnce({ count: 1 });

    const res = await deleteReq("rec-1");
    expect(res.status).toBe(200);
    expect(prismaMock.projectRecord.deleteMany).toHaveBeenCalledWith({
      where: { id: "rec-1", userId: "user-1" },
    });
  });
});
