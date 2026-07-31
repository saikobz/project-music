import Omise from "omise";
import https from "https";
import { URL, urlToHttpOptions } from "url";

export const omise = Omise({
  publicKey: process.env.NEXT_PUBLIC_OMISE_PUBLIC_KEY || "",
  secretKey: process.env.OMISE_SECRET_KEY || "",
});

// omise@1.1.0 (ผ่าน https-proxy-agent/agent-base) patch https.request แบบ global
// โดยไม่รองรับรูปแบบ https.request(url, options) ที่ openid-client ใช้ตอน
// ค้นหา OpenID configuration จึงเกิด TypeError "listener argument must be function"
// wrapper นี้แปลง url + options ให้เป็น options object เดียว ทำให้ทั้งสองทำงานร่วมกัน
const originalRequest = https.request;

https.request = function (
  input: string | URL | https.RequestOptions,
  optionsOrCallback?: https.RequestOptions | ((res: any) => void),
  callback?: (res: any) => void
) {
  if (typeof input === "string" && optionsOrCallback && typeof optionsOrCallback === "object") {
    return originalRequest({ ...urlToHttpOptions(new URL(input)), ...optionsOrCallback }, callback as any);
  }
  return originalRequest(input as https.RequestOptions, optionsOrCallback as any);
} as typeof https.request;
