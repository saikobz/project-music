jest.mock("@next-auth/prisma-adapter", () => ({
  PrismaAdapter: jest.fn(),
}));

jest.mock("next-auth", () => ({
  __esModule: true,
  default: jest.fn(() => jest.fn()),
  getServerSession: jest.fn(),
}));

jest.mock("@/lib/omise", () => ({
  omise: {
    charges: {
      create: jest.fn().mockResolvedValue({
        id: "chrg_test_123",
        source: {
          scannable_code: {
            image: { download_uri: "https://api.omise.co/qr/test.png" },
          },
        },
      }),
    },
    customers: {
      create: jest.fn().mockResolvedValue({ id: "cust_test_123" }),
      update: jest.fn().mockResolvedValue({ id: "cust_test_123" }),
    },
    schedules: {
      create: jest.fn().mockResolvedValue({ id: "schd_test_123" }),
      destroy: jest.fn().mockResolvedValue({ id: "schd_old" }),
    },
  },
}));

jest.mock("@/lib/prisma", () => ({
  prisma: {
    user: { findUnique: jest.fn(), update: jest.fn() },
    subscription: { upsert: jest.fn() },
  },
}));

import { POST } from "../app/api/subscription/checkout/route";
import { getServerSession } from "next-auth";
import { omise } from "@/lib/omise";
import { prisma } from "@/lib/prisma";

const prismaMock = prisma as unknown as {
  user: { findUnique: jest.Mock; update: jest.Mock };
  subscription: { upsert: jest.Mock };
};

describe("Checkout API Endpoint", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should return 401 if user is unauthenticated", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce(null);

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({ tier: "BASIC", paymentMethod: "PROMPTPAY" }),
    });

    const res = await POST(req);
    expect(res.status).toBe(401);
  });

  it("should return PromptPay QR code for authenticated user", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({ tier: "BASIC", paymentMethod: "PROMPTPAY" }),
    });

    const res = await POST(req);
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.success).toBe(true);
    expect(data.qrCodeUrl).toBe("https://api.omise.co/qr/test.png");
  });

  it("should include userId/tier metadata in scheduled charge (F5)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });
    prismaMock.user.findUnique.mockResolvedValueOnce(null);

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({
        tier: "BASIC",
        paymentMethod: "CREDIT_CARD",
        cardToken: "tokn_test_123",
      }),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);

    // schedule ต้องถูกสร้างพร้อม metadata เพื่อให้ webhook ผูก charge รายเดือนกับ user ได้
    expect(omise.schedules.create).toHaveBeenCalledWith(
      expect.objectContaining({
        charge: expect.objectContaining({
          customer: "cust_test_123",
          amount: 9900,
          metadata: { userId: "user-123", tier: "BASIC" },
        }),
      })
    );
  });

  it("destroys existing schedule before creating a new one (F3)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-123",
      omiseCustomerId: "cust_existing",
      subscription: { omiseScheduleId: "schd_old" },
    });

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({
        tier: "PRO",
        paymentMethod: "CREDIT_CARD",
        cardToken: "tokn_test_123",
      }),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(omise.schedules.destroy).toHaveBeenCalledWith("schd_old");
    expect(omise.schedules.create).toHaveBeenCalled();
  });

  it("attaches new card to existing customer instead of ignoring it (F7)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-123",
      omiseCustomerId: "cust_existing",
      subscription: null,
    });

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({
        tier: "BASIC",
        paymentMethod: "CREDIT_CARD",
        cardToken: "tokn_new_card",
      }),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(omise.customers.update).toHaveBeenCalledWith("cust_existing", { card: "tokn_new_card" });
    expect(omise.customers.create).not.toHaveBeenCalled();
  });

  it("creates subscription with PENDING status until first successful charge (F6)", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });
    prismaMock.user.findUnique.mockResolvedValueOnce({
      id: "user-123",
      omiseCustomerId: null,
      subscription: null,
    });

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({
        tier: "BASIC",
        paymentMethod: "CREDIT_CARD",
        cardToken: "tokn_test_123",
      }),
    });

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.subscription.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        update: expect.objectContaining({ tier: "BASIC", status: "PENDING" }),
        create: expect.objectContaining({ tier: "BASIC", status: "PENDING" }),
      })
    );
  });

  it("rejects invalid tier with 400", async () => {
    (getServerSession as jest.Mock).mockResolvedValueOnce({
      user: { id: "user-123", email: "test@example.com", tier: "FREE" },
    });

    const req = new Request("http://localhost:3000/api/subscription/checkout", {
      method: "POST",
      body: JSON.stringify({ tier: "HACKER_TIER", paymentMethod: "PROMPTPAY" }),
    });

    const res = await POST(req);
    expect(res.status).toBe(400);
  });
});
