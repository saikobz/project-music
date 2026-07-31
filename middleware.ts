// middleware.ts
// จำกัดอัตราการเรียก auth endpoints (login, register, เปลี่ยนรหัสผ่าน) ตาม IP
// กัน brute force — ใช้ใน-memory bucket (per-process, reset เมื่อ restart)
import { NextRequest, NextResponse } from "next/server";
import { checkRateLimit, getClientIp } from "@/lib/rate-limit";

// จำกัด 10 ครั้ง/นาที ต่อ IP ต่อ endpoint
const RATE_LIMIT = 10;
const RATE_WINDOW_MS = 60 * 1000;

const AUTH_PATHS = new Set([
  "/api/auth/callback/credentials",
  "/api/auth/register",
  "/api/account/password",
]);

export function middleware(req: NextRequest) {
  if (req.method !== "POST") {
    return NextResponse.next();
  }

  const path = req.nextUrl.pathname;
  if (!AUTH_PATHS.has(path)) {
    return NextResponse.next();
  }

  const ip = getClientIp(req);
  if (!checkRateLimit(`${ip}:${path}`, RATE_LIMIT, RATE_WINDOW_MS)) {
    return NextResponse.json(
      { error: "Too many requests. Please try again later." },
      { status: 429 }
    );
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/api/auth/:path*", "/api/account/password"],
};
