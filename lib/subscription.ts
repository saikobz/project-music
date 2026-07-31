// lib/subscription.ts
// ศูนย์กลางสำหรับ: สิทธิ์ tier ที่มีผลจริง, โควตาต่อ tier, ราคา tier
// (กัน bug แบบ F2: ผู้ใช้ยกเลิก/หมดอายุแล้วยังได้สิทธิ์ PRO ตลอดกาล
//  และ DRY: ค่า quota/ราคาไม่ซ้ำกระจายในหลายไฟล์ — C4/C5)

export type SubscriptionLike = {
  tier: string | null;
  status: string | null;
  currentPeriodEnd: Date | string | null;
} | null;

// โควตาประมวลผลรายเดือนต่อ tier (PRO = -1 หมายถึงไม่จำกัด)
export const TIER_MONTHLY_QUOTA: Record<string, number> = {
  FREE: 3,
  BASIC: 15,
  PRO: -1,
};

// ราคาต่อเดือน (หน่วย: สตางค์ THB*100)
export const TIER_PRICES: Record<string, number> = {
  BASIC: 9900,
  PRO: 29900,
};

export function getMonthlyQuota(tier: string): number {
  return TIER_MONTHLY_QUOTA[tier] ?? TIER_MONTHLY_QUOTA.FREE;
}

export function getTierPrice(tier: string): number | null {
  return TIER_PRICES[tier] ?? null;
}

export function getEffectiveTier(subscription: SubscriptionLike, now: Date = new Date()): string {
  if (!subscription) return "FREE";
  if (subscription.status !== "ACTIVE") return "FREE";
  const periodEnd = subscription.currentPeriodEnd;
  if (periodEnd && new Date(periodEnd) <= now) return "FREE";
  return subscription.tier || "FREE";
}
