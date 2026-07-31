import https from "https";

describe("Omise https.request compatibility", () => {
  it("supports https.request(url, options, cb) after loading lib/omise", async () => {
    const mockRequest = jest.fn();
    https.request = mockRequest as any;

    // https เป็น node module ที่แชร์กันใน worker เดียว — ลบ marker ของ L5
    // (ถ้า test file ก่อนหน้าเคยโหลด lib/omise จริง จะข้ามการ patch ทำให้ test นี้พัง)
    delete (https as unknown as Record<string, unknown>)["__harmoniqHttpsPatched"];

    await import("../lib/omise");

    const cb = jest.fn();
    (https.request as any)("https://example.com/x", { method: "GET" }, cb);

    expect(mockRequest).toHaveBeenCalledTimes(1);
    const [options, listener] = mockRequest.mock.calls[0];
    expect(typeof options).toBe("object");
    expect((options as any).hostname).toBe("example.com");
    expect(typeof listener).toBe("function");
  });
});
