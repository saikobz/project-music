import { NextAuthOptions, getServerSession, type Session } from "next-auth";
import { PrismaAdapter } from "@next-auth/prisma-adapter";
import GoogleProvider from "next-auth/providers/google";
import FacebookProvider from "next-auth/providers/facebook";
import LineProvider from "next-auth/providers/line";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { PERIOD_MS } from "@/lib/config";

// ใช้สำหรับเทียบ bcrypt เมื่อไม่พบผู้ใช้ (กัน timing attack แยกแยะว่ามี email นี้ในระบบ)
const DUMMY_PASSWORD_HASH = bcrypt.hashSync("dummy-password-for-timing", 10);

// แยกเป็นฟังก์ชันที่ test ได้โดยตรง (NextAuth ครอบ authorize ไว้ไม่ให้เรียกตรง)
export async function credentialsAuthorize(
  credentials: Record<string, string> | undefined
): Promise<{ id: string; email: string; name: string | null; image: string | null } | null> {
  if (!credentials?.email || !credentials?.password) {
    throw new Error("Email and password are required");
  }

  // normalize email ให้ตรงกับการลงทะเบียน (กัน email ตัวพิมพ์ต่างกันหาบัญชีไม่เจอ)
  const email = credentials.email.trim().toLowerCase();

  const user = await prisma.user.findUnique({
    where: { email },
  });

  // F9: ใช้ข้อความ error เดียวกันทั้ง "ไม่มี email นี้" และ "รหัสผ่านผิด"
  // เพื่อไม่ให้โจมตี enumerate ว่า email ใดลงทะเบียนไว้แล้ว + เทียบ bcrypt เสมอ (กัน timing)
  const passwordHash = user?.password || DUMMY_PASSWORD_HASH;
  const isValid = await bcrypt.compare(credentials.password, passwordHash);
  if (!user || !user.password || !isValid) {
    throw new Error("Invalid email or password");
  }

  return { id: user.id, email: user.email, name: user.name, image: user.image };
}

// ตัวช่วยยืนยัน session สำหรับ API routes (C9: ลดโค้ดซ้ำ getServerSession + 401 ในทุก route)
export async function requireSession(): Promise<
  | { session: Session; response: null }
  | { session: null; response: NextResponse }
> {
  const session = await getServerSession(authOptions);
  if (!session?.user) {
    return {
      session: null,
      response: NextResponse.json({ error: "Unauthorized" }, { status: 401 }),
    };
  }
  return { session, response: null };
}

// ตัวช่วยยืนยันตัวตนสำหรับ action ที่ทำลายข้อมูล (ลบบัญชี / ยกเลิก subscription — C10)
// - ผู้ใช้ที่มี password: ต้องยืนยันด้วย password
// - ผู้ใช้ OAuth-only: ต้อง re-authenticate ผ่าน provider ล่าสุด (M19+ — reauthAt ใน session
//   ถูก stamp ตอน login ใหม่; ตรวจไม่เกิน 5 นาที)
export async function verifyAccountAuth(
  user: { password: string | null; email: string },
  password: string | undefined,
  confirmEmail: string | undefined,
  reauthAt?: number
): Promise<{ status: number; error: string } | null> {
  if (user.password) {
    if (!password) {
      return { status: 400, error: "Password is required" };
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return { status: 401, error: "Incorrect password" };
    }
    return null;
  }

  // OAuth-only: ต้อง re-auth ผ่าน provider ภายใน 5 นาที (M19+)
  const REAUTH_WINDOW_MS = 5 * 60 * 1000;
  if (typeof reauthAt !== "number" || Date.now() - reauthAt > REAUTH_WINDOW_MS) {
    return {
      status: 403,
      error: "กรุณายืนยันตัวตนผ่านบัญชีที่เชื่อมต่ออีกครั้งก่อนดำเนินการ",
    };
  }
  return null;
}

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  session: { strategy: "jwt" },
  pages: {
    signIn: "/auth/signin",
  },
  logger: {
    // ดีบั๊ก OAuth error: แสดง code + message จริงของ NextAuth (ไม่ใช่แค่ error generic)
    error(code, ...message) {
      console.error("NextAuth error:", code, ...message);
    },
  },
  providers: [
    CredentialsProvider({
      id: "credentials",
      name: "Email & Password",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        return credentialsAuthorize(credentials);
      },
    }),
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    }),
    FacebookProvider({
      clientId: process.env.FACEBOOK_CLIENT_ID || "",
      clientSecret: process.env.FACEBOOK_CLIENT_SECRET || "",
      authorization: {
        // L7: ขอแค่ public_profile — อย่าขอ scope email เพราะ Facebook ปฏิเสธ
        // (ต้องผ่าน App Review) ทำให้ login ผ่านไม่ได้
        // email ที่ได้มาเป็น null จะใช้ fallback fb_{id}@facebook.local ใน profile() แทน
        params: { scope: "public_profile" },
      },
      profile(profile) {
        return {
          id: profile.id,
          name: profile.name,
          email: profile.email || `fb_${profile.id}@facebook.local`,
          image: profile.picture?.data?.url || null,
        };
      },
    }),
    LineProvider({
      clientId: process.env.LINE_CLIENT_ID || "",
      clientSecret: process.env.LINE_CLIENT_SECRET || "",
      profile(profile) {
        return {
          id: profile.sub,
          name: profile.name,
          // LINE จะส่ง email ก็ต่อเมื่อได้รับอนุญาตจากผู้ใช้ จึงใช้ fallback เป็น id ที่ได้จาก LINE
          email: profile.email || `line_${profile.sub}@line.local`,
          image: profile.picture,
        };
      },
    }),
  ],
  callbacks: {
    async signIn({ user, account }) {
      // สร้าง Subscription และ UsageQuota (FREE 3 เพลง) ให้ผู้ใช้ที่สมัครผ่าน OAuth
      // L2: ใช้ upsert แทน findUnique-then-create (กัน race เมื่อ sign-in พร้อมกัน -> แถวซ้ำ)
      if (account && account.provider !== "credentials" && user?.id) {
        try {
          await prisma.subscription.upsert({
            where: { userId: user.id },
            update: {},
            create: { userId: user.id, tier: "FREE", status: "ACTIVE" },
          });
          const periodStart = new Date();
          await prisma.usageQuota.upsert({
            where: { userId_periodStart: { userId: user.id, periodStart } },
            update: {},
            create: {
              userId: user.id,
              monthlyQuota: 3,
              usedCount: 0,
              periodStart,
              periodEnd: new Date(Date.now() + PERIOD_MS),
            },
          });
        } catch (err) {
          console.error("Failed to initialize quota for OAuth user:", err);
        }
      }
      return true;
    },
    async jwt({ token, user }) {
      if (user) {
        const dbUser = await prisma.user.findUnique({
          where: { id: user.id },
          include: { subscription: true },
        });
        token.id = user.id;
        token.tier = dbUser?.subscription?.tier || "FREE";
        token.omiseCustomerId = dbUser?.omiseCustomerId || undefined;
        // M19+: stamp เวลา re-auth ทุกครั้งที่มีการ sign-in ใหม่
        // (ใช้ตรวจสอบ destructive action สำหรับผู้ใช้ OAuth-only)
        token.reauthAt = Date.now();
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
        // L1: อ่าน tier ล่าสุดจาก DB ทุกครั้งที่สร้าง session
        // (เดิมค่าใน JWT stale หลังอัปเกรด/เปลี่ยนแพ็กเกจ -> UI แสดง tier เก่าจนกว่า re-login)
        if (token.id) {
          try {
            const dbUser = await prisma.user.findUnique({
              where: { id: token.id as string },
              include: { subscription: true },
            });
            session.user.tier = dbUser?.subscription?.tier || "FREE";
            // ใช้กับหน้า confirm-delete: รู้ว่าผู้ใช้ OAuth-only หรือไม่ (ไม่ส่ง hash กลับ)
            session.user.hasPassword = !!dbUser?.password;
          } catch {
            session.user.tier = (token.tier as string) || "FREE";
            session.user.hasPassword = true; // ไม่รู้ -> ถือว่ามี password (fallback ปลอดภัย)
          }
        } else {
          session.user.tier = (token.tier as string) || "FREE";
          session.user.hasPassword = true;
        }
        session.user.omiseCustomerId = token.omiseCustomerId as string;
        session.user.reauthAt = token.reauthAt as number | undefined;
      }
      return session;
    },
  },
};
