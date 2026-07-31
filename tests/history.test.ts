jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

import { GET, POST } from "../app/api/history/route";
import { getServerSession } from "next-auth";

describe("History API — GET", () => {
  it("should return 401 if unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);
    const res = await GET();
    expect(res.status).toBe(401);
  });
});

describe("History API — POST", () => {
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
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-1", email: "test@test.com" },
    });
    const req = new Request("http://localhost/api/history", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const res = await POST(req);
    expect(res.status).toBe(400);
  });
});
