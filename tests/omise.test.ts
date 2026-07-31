import https from "https";

describe("Omise https.request compatibility", () => {
  it("supports https.request(url, options, cb) after loading lib/omise", async () => {
    const mockRequest = jest.fn();
    https.request = mockRequest as any;

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
