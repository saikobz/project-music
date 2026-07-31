import { NextAuthOptions } from "next-auth";
import { PrismaAdapter } from "@auth/prisma-adapter";
import GoogleProvider from "next-auth/providers/google";
import FacebookProvider from "next-auth/providers/facebook";
import LineProvider from "next-auth/providers/line";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { prisma } from "@/lib/prisma";

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  session: { strategy: "jwt" },
  pages: {
    signIn: "/auth/signin",
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
        if (!credentials?.email || !credentials?.password) {
          throw new Error("Email and password are required");
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email },
        });

        if (!user || !user.password) {
          throw new Error("No user found with this email");
        }

        const isValid = await bcrypt.compare(credentials.password, user.password);
        if (!isValid) {
          throw new Error("Incorrect password");
        }

        return { id: user.id, email: user.email, name: user.name, image: user.image };
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
      // ถ้ายังไม่มี record อยู่ก่อน (ผู้ใช้ที่สมัครผ่าน Credentials จะถูกสร้างไว้แล้วใน register route)
      if (account && account.provider !== "credentials" && user?.id) {
        try {
          const existing = await prisma.user.findUnique({
            where: { id: user.id },
            include: { subscription: true },
          });
          if (existing && !existing.subscription) {
            await prisma.subscription.create({
              data: { userId: existing.id, tier: "FREE", status: "ACTIVE" },
            });
            await prisma.usageQuota.create({
              data: {
                userId: existing.id,
                monthlyQuota: 3,
                usedCount: 0,
                periodStart: new Date(),
                periodEnd: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
              },
            });
          }
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
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
        session.user.tier = (token.tier as string) || "FREE";
        session.user.omiseCustomerId = token.omiseCustomerId as string;
      }
      return session;
    },
  },
};
