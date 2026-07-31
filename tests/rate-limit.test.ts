/**
 * regression tests สำหรับ TU-9 (F9): rate limiter (in-memory)
 * - เกิน limit -> ถูกบล็อก
 * - หมด window -> เริ่มนับใหม่
 *
 * @jest-environment node
 */
import { checkRateLimit, getClientIp } from "../lib/rate-limit";

describe("Rate Limit (TU-9 F9)", () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it("allows requests up to the limit then blocks", () => {
    const key = "ip:/api/auth/register";
    for (let i = 0; i < 10; i++) {
      expect(checkRateLimit(key, 10, 60_000)).toBe(true);
    }
    expect(checkRateLimit(key, 10, 60_000)).toBe(false);
  });

  it("resets after the window expires", () => {
    const key = "ip:/api/auth/register";
    for (let i = 0; i < 10; i++) {
      checkRateLimit(key, 10, 60_000);
    }
    expect(checkRateLimit(key, 10, 60_000)).toBe(false);

    jest.advanceTimersByTime(60_001);
    expect(checkRateLimit(key, 10, 60_000)).toBe(true);
  });

  it("tracks different keys independently", () => {
    expect(checkRateLimit("ip1:/path", 1, 60_000)).toBe(true);
    expect(checkRateLimit("ip1:/path", 1, 60_000)).toBe(false);
    expect(checkRateLimit("ip2:/path", 1, 60_000)).toBe(true);
  });

  it("extracts client IP from x-forwarded-for header", () => {
    const req = new Request("http://localhost/", {
      headers: { "x-forwarded-for": "203.0.113.7, 10.0.0.1" },
    });
    expect(getClientIp(req)).toBe("203.0.113.7");
  });

  it("falls back to unknown when no IP header present", () => {
    const req = new Request("http://localhost/");
    expect(getClientIp(req)).toBe("unknown");
  });
});
