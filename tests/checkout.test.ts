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
      create: jest.fn(),
    },
    schedules: {
      create: jest.fn(),
    },
  },
}));

import { POST } from "../app/api/subscription/checkout/route";
import { getServerSession } from "next-auth";

describe("Checkout API Endpoint", () => {
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
});
