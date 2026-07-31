// lib/config.ts
// ศูนย์กลางค่าคงที่และการตั้งค่าระบบสำหรับ Frontend

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100MB

// ระยะเวลา subscription/quota หนึ่งรอบ (30 วัน)
export const PERIOD_MS = 30 * 24 * 60 * 60 * 1000;

// รายการ action การประมวลผลเสียง (C8/C14: ใช้เป็น source of truth เดียว)
export const AUDIO_ACTIONS = ["separate", "eq-ai", "compressor", "pitch"] as const;
export type AudioAction = (typeof AUDIO_ACTIONS)[number];

// แผนที่ action ฝั่ง UI -> ชื่อ endpoint/action ที่ backend ใช้ (กัน naming หลุด 3 ระบบ)
export const ACTION_TO_BACKEND: Record<AudioAction, string> = {
  separate: "separate",
  "eq-ai": "apply-eq-ai",
  compressor: "apply-compressor",
  pitch: "pitch-shift",
};

export const DEFAULT_STEMS = ["vocals", "drums", "bass", "other"] as const;

export const DEFAULT_GENRES = ["pop", "rock", "trap", "country", "soul"] as const;

export const OAUTH_PROVIDERS = [
  { id: "google", name: "Google", icon: "google" },
  { id: "facebook", name: "Facebook", icon: "facebook" },
  { id: "line", name: "LINE", icon: "line" },
] as const;

export type StemType = typeof DEFAULT_STEMS[number];
export type GenreType = typeof DEFAULT_GENRES[number];
