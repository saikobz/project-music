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

// L5: กัน patch ซ้อนเมื่อ HMR reload (ถ้า patch ไปแล้วจะไม่ wrapper ซ้ำอีก)
const HARMONIQ_PATCH_MARKER = "__harmoniqHttpsPatched";

if (!(https as unknown as Record<string, unknown>)[HARMONIQ_PATCH_MARKER]) {
  https.request = function (
    input: string | URL | https.RequestOptions,
    optionsOrCallback?: https.RequestOptions | ((res: any) => void),
    callback?: (res: any) => void
  ) {
    if (optionsOrCallback && typeof optionsOrCallback === "object") {
      // รองรับทั้ง string URL และ URL object + options
      const url = typeof input === "string" ? new URL(input) : input;
      if (url instanceof URL) {
        return originalRequest(
          { ...urlToHttpOptions(url), ...optionsOrCallback },
          callback as any
        );
      }
    }
    return originalRequest(input as https.RequestOptions, optionsOrCallback as any);
  } as typeof https.request;

  (https as unknown as Record<string, unknown>)[HARMONIQ_PATCH_MARKER] = true;
}
