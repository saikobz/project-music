// lib/rate-limit.ts
// ตัวจำกัดอัตราการเรียก API แบบ in-memory (per-process) สำหรับ auth endpoints
// กัน brute force ที่ login/register/เปลี่ยนรหัสผ่าน
// หมายเหตุ: ใช้ใน middleware (edge runtime) — reset ทุกครั้งที่ restart server

type Entry = { count: number; resetAt: number };

const buckets = new Map<string, Entry>();

export function checkRateLimit(key: string, limit: number, windowMs: number): boolean {
  const now = Date.now();

  // กวาด entry ที่หมดอายุเป็นระยะ เพื่อไม่ให้ Map โตไม่มีที่สิ้นสุด
  if (buckets.size > 1000) {
    for (const [k, entry] of buckets) {
      if (entry.resetAt <= now) {
        buckets.delete(k);
      }
    }
  }

  const entry = buckets.get(key);
  if (!entry || entry.resetAt <= now) {
    buckets.set(key, { count: 1, resetAt: now + windowMs });
    return true;
  }
  entry.count += 1;
  return entry.count <= limit;
}

export function getClientIp(req: Request): string {
  return req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() || "unknown";
}
