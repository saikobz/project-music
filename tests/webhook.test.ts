/**
 * regression tests สำหรับ TU-4 (Webhook Security):
 * - F1: ต้อง fail-closed — ไม่มี secret -> 503, ไม่ส่ง signature -> 403, signature ผิด -> 403
 * - F8: charge.complete ที่ล้มเหลวต้องบันทึกเป็น "failed" และต้องไม่ activate subscription
 * - verify กับ raw body (req.text()) ไม่ใช่ JSON.stringify
 *
 * @jest-environment node
 */
import crypto from "crypto";

jest.mock("@/lib/prisma", () => ({
  prisma: {
    paymentRecord: { create: jest.fn() },
    subscription: { upsert: jest.fn(), update: jest.fn() },
    usageQuota: { create: jest.fn(), deleteMany: jest.fn() },
  },
}));

jest.mock("@/lib/omise", () => ({
  omise: {
    events: { retrieve: jest.fn() },
  },
}));

import { prisma } from "@/lib/prisma";
import { omise } from "@/lib/omise";
import { POST } from "../app/api/webhooks/omise/route";

const SECRET = "test-webhook-secret";
const prismaMock = prisma as unknown as {
  paymentRecord: { create: jest.Mock };
  subscription: { upsert: jest.Mock; update: jest.Mock };
  usageQuota: { create: jest.Mock; deleteMany: jest.Mock };
};

function sign(rawBody: string): string {
  return crypto.createHmac("sha256", SECRET).update(rawBody).digest("hex");
}

function buildRequest(payload: unknown, signature?: string): Request {
  const rawBody = JSON.stringify(payload);
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (signature !== undefined) {
    headers["x-omise-signature"] = signature;
  }
  return new Request("http://localhost:3000/api/webhooks/omise", {
    method: "POST",
    body: rawBody,
    headers,
  });
}

describe("Omise Webhook (TU-4)", () => {
  const originalSecret = process.env.OMISE_WEBHOOK_SECRET;

  beforeEach(() => {
    process.env.OMISE_WEBHOOK_SECRET = SECRET;
    jest.clearAllMocks();
  });

  afterAll(() => {
    if (originalSecret === undefined) {
      delete process.env.OMISE_WEBHOOK_SECRET;
    } else {
      process.env.OMISE_WEBHOOK_SECRET = originalSecret;
    }
  });

  it("rejects request with 503 when webhook secret is not configured (F1)", async () => {
    delete process.env.OMISE_WEBHOOK_SECRET;
    delete process.env.OMISE_SECRET_KEY;
    const req = buildRequest({ key: "charge.complete", data: {} });

    const res = await POST(req);
    expect(res.status).toBe(503);
  });

  it("verifies event id via Omise API when no signature secret is set (G1/G3)", async () => {
    // จำลอง Omise ที่ไม่ส่ง HMAC signature: ใช้ OMISE_SECRET_KEY + events.retrieve ยืนยันแทน
    delete process.env.OMISE_WEBHOOK_SECRET;
    process.env.OMISE_SECRET_KEY = "skm_test_123";
    (omise.events.retrieve as jest.Mock).mockResolvedValueOnce({ id: "evnt_test_1" });

    const payload = {
      key: "charge.complete",
      data: {
        id: "evnt_test_1",
        status: "successful",
        amount: 9900,
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };
    const req = buildRequest(payload); // ไม่มี signature header

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(omise.events.retrieve).toHaveBeenCalledWith("evnt_test_1");
    expect(prismaMock.subscription.upsert).toHaveBeenCalled();
  });

  it("rejects forged event id when verifying via Omise API (G1/G3)", async () => {
    delete process.env.OMISE_WEBHOOK_SECRET;
    process.env.OMISE_SECRET_KEY = "skm_test_123";
    (omise.events.retrieve as jest.Mock).mockRejectedValueOnce(new Error("not found"));

    const payload = { key: "charge.complete", data: { id: "evnt_forged_1" } };
    const req = buildRequest(payload);

    const res = await POST(req);
    expect(res.status).toBe(403);
    expect(prismaMock.subscription.upsert).not.toHaveBeenCalled();
  });

  it("rejects event id that does not look like an Omise event (G1/G3)", async () => {
    delete process.env.OMISE_WEBHOOK_SECRET;
    process.env.OMISE_SECRET_KEY = "skm_test_123";

    const payload = { key: "charge.complete", data: { id: "not-an-event" } };
    const req = buildRequest(payload);

    const res = await POST(req);
    expect(res.status).toBe(403);
    expect(omise.events.retrieve).not.toHaveBeenCalled();
  });

  it("rejects with 503 when no signature and no verification key configured", async () => {
    // มี secret แต่ไม่มี signature และไม่มี OMISE_SECRET_KEY -> ไม่มีวิธียืนยัน -> 503
    delete process.env.OMISE_SECRET_KEY;
    const req = buildRequest({ key: "charge.complete", data: {} });

    const res = await POST(req);
    expect(res.status).toBe(503);
  });

  it("rejects request with 403 when signature is invalid (F1)", async () => {
    const req = buildRequest({ key: "charge.complete", data: {} }, "wrong-signature");

    const res = await POST(req);
    expect(res.status).toBe(403);
  });

  it("rejects with 403 when signature was computed over different raw body (F1)", async () => {
    // เซ็นด้วย body คนละ format กับที่ส่งจริง -> ต้องโดน reject
    const payload = { key: "charge.complete", data: { id: "chrg_1" } };
    const wrongSig = crypto
      .createHmac("sha256", SECRET)
      .update(JSON.stringify({ key: "charge.complete", data: { id: "chrg_2" } }))
      .digest("hex");
    const req = buildRequest(payload, wrongSig);

    const res = await POST(req);
    expect(res.status).toBe(403);
  });

  it("processes charge.complete successful: records payment and activates subscription", async () => {
    const payload = {
      key: "charge.complete",
      data: {
        id: "chrg_test_123",
        status: "successful",
        amount: 29900,
        metadata: { userId: "user-123", tier: "PRO" },
      },
    };
    const req = buildRequest(payload, sign(JSON.stringify(payload)));

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.paymentRecord.create).toHaveBeenCalledWith({
      data: expect.objectContaining({
        userId: "user-123",
        omiseChargeId: "chrg_test_123",
        amount: 29900,
        status: "successful",
      }),
    });
    expect(prismaMock.subscription.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        where: { userId: "user-123" },
        update: expect.objectContaining({ tier: "PRO", status: "ACTIVE" }),
      })
    );
    // L8: ล้าง quota รอบเก่าก่อนสร้างรอบใหม่ (ไม่ให้ตารางบาน)
    expect(prismaMock.usageQuota.deleteMany).toHaveBeenCalledWith({ where: { userId: "user-123" } });
    expect(prismaMock.usageQuota.create).toHaveBeenCalledTimes(1);
  });

  it("records failed charge as failed and does NOT activate subscription (F8)", async () => {
    const payload = {
      key: "charge.complete",
      data: {
        id: "chrg_test_fail",
        status: "failed",
        amount: 9900,
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };
    const req = buildRequest(payload, sign(JSON.stringify(payload)));

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.paymentRecord.create).toHaveBeenCalledWith({
      data: expect.objectContaining({ omiseChargeId: "chrg_test_fail", status: "failed" }),
    });
    // ไม่มีการ activate subscription / สร้าง usageQuota
    expect(prismaMock.subscription.upsert).not.toHaveBeenCalled();
    expect(prismaMock.usageQuota.create).not.toHaveBeenCalled();
  });

  it("returns 200 and touches no DB for unknown events", async () => {
    const payload = { key: "charge.unknown_event", data: { id: "x" } };
    const req = buildRequest(payload, sign(JSON.stringify(payload)));

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.paymentRecord.create).not.toHaveBeenCalled();
    expect(prismaMock.subscription.upsert).not.toHaveBeenCalled();
    expect(prismaMock.subscription.update).not.toHaveBeenCalled();
    expect(prismaMock.usageQuota.create).not.toHaveBeenCalled();
  });

  it("processes schedule.process successful using metadata from scheduled charge (F5)", async () => {
    const payload = {
      key: "schedule.process",
      data: {
        status: "successful",
        charge: "chrg_schedule_1",
        amount: 9900,
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };
    const req = buildRequest(payload, sign(JSON.stringify(payload)));

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.paymentRecord.create).toHaveBeenCalledWith({
      data: expect.objectContaining({ omiseChargeId: "chrg_schedule_1", status: "successful" }),
    });
    expect(prismaMock.subscription.upsert).toHaveBeenCalled();
  });

  it("marks subscription past due when scheduled charge fails (F5)", async () => {
    const payload = {
      key: "schedule.process",
      data: {
        status: "failed",
        charge: "chrg_schedule_fail",
        amount: 9900,
        metadata: { userId: "user-123", tier: "BASIC" },
      },
    };
    const req = buildRequest(payload, sign(JSON.stringify(payload)));

    const res = await POST(req);
    expect(res.status).toBe(200);
    expect(prismaMock.subscription.update).toHaveBeenCalledWith({
      where: { userId: "user-123" },
      data: { status: "PAST_DUE" },
    });
  });
});
