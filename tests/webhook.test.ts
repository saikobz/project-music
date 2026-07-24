jest.mock("@/lib/prisma", () => ({
  prisma: {
    subscription: { upsert: jest.fn() },
    usageQuota: { create: jest.fn() },
  },
}));

import { POST } from "../app/api/webhooks/omise/route";

describe("Omise Webhook Test", () => {
  it("should process charge.complete event for PromptPay", async () => {
    const payload = {
      key: "charge.complete",
      data: {
        id: "chrg_test_123",
        status: "successful",
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };

    const req = new Request("http://localhost:3000/api/webhooks/omise", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    const res = await POST(req);
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.received).toBe(true);
  });
});
