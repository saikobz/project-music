import { DefaultSession } from "next-auth";

declare module "next-auth" {
  interface Session {
    user: {
      id: string;
      tier: string;
      omiseCustomerId?: string;
      // M19+: timestamp ของการ re-auth ครั้งล่าสุด (ตรวจ destructive action)
      reauthAt?: number;
      // ผู้ใช้มี password หรือไม่ (ใช้กับหน้า confirm-delete; ไม่ส่ง hash กลับ)
      hasPassword?: boolean;
    } & DefaultSession["user"];
  }
}

// JWT type จริงของ NextAuth v4 อยู่ที่ module "next-auth/jwt" (C13)
declare module "next-auth/jwt" {
  interface JWT {
    id?: string;
    tier?: string;
    omiseCustomerId?: string;
    reauthAt?: number;
  }
}
