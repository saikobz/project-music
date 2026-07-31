// lib/config.ts
// ศูนย์กลางค่าคงที่และการตั้งค่าระบบสำหรับ Frontend

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const MAX_UPLOAD_BYTES = 100 * 1024 * 1024; // 100MB

export const DEFAULT_STEMS = ["vocals", "drums", "bass", "other"] as const;

export const DEFAULT_GENRES = ["pop", "rock", "trap", "country", "soul"] as const;

export const OAUTH_PROVIDERS = [
  { id: "google", name: "Google", icon: "google" },
  { id: "facebook", name: "Facebook", icon: "facebook" },
  { id: "line", name: "LINE", icon: "line" },
] as const;

export type StemType = typeof DEFAULT_STEMS[number];
export type GenreType = typeof DEFAULT_GENRES[number];
