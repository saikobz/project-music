"use client";
import React, { useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { Navbar } from "@/app/components/Navbar";
import { Footer } from "@/app/components/Footer";
import PasswordInput from "@/app/components/PasswordInput";
import SocialButtons from "@/app/components/SocialButtons";

function SignInForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlError = searchParams?.get("error");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (urlError) {
      if (urlError === "OAuthSignin") {
        setError(
          "ไม่สามารถเชื่อมต่อกับผู้ให้บริการ (Google/Facebook/LINE) ได้ กรุณาลองใหม่อีกครั้ง — หากยังคงเกิดซ้ำ ให้ตรวจสอบ Client ID/Secret และ Redirect URI ในการตั้งค่าของผู้ให้บริการ"
        );
      } else if (urlError === "CredentialsSignin") {
        setError("อีเมลหรือรหัสผ่านไม่ถูกต้อง");
      } else {
        setError("เกิดข้อผิดพลาดในการเข้าสู่ระบบ");
      }
    }
  }, [urlError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const result = await signIn("credentials", { email, password, redirect: false });
    setLoading(false);
    if (result?.error) {
      setError("อีเมลหรือรหัสผ่านไม่ถูกต้อง");
    } else {
      router.push("/");
    }
  };

  return (
    <div className="bg-[#161412] border border-[#2C2824] rounded-2xl p-8 max-w-md w-full shadow-2xl space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Welcome Back</h1>
        <p className="text-xs text-[#8E8E8E]">Sign in to HarmoniQ — AI Music Separator & Audio Toolkit</p>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg">{error}</div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Email Address</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@example.com"
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#F97316] transition"
          />
        </div>
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Password</label>
          <PasswordInput value={password} onChange={setPassword} />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-bold text-sm rounded-lg transition cursor-pointer"
        >
          {loading ? "Processing..." : "Sign In"}
        </button>
      </form>

      <div className="relative flex items-center justify-center border-t border-[#2C2824] pt-4">
        <span className="bg-[#161412] px-2 text-[10px] uppercase text-[#8E8E8E] font-bold">Or continue with</span>
      </div>

      <SocialButtons />

      <p className="text-center text-xs text-[#8E8E8E]">
        ยังไม่มีบัญชี?{" "}
        <Link href="/auth/signup" className="text-[#F97316] font-semibold hover:underline">
          Create Account
        </Link>
      </p>
    </div>
  );
}

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow flex items-center justify-center p-4 py-12">
        <Suspense fallback={<div className="text-sm text-[#8E8E8E]">Loading...</div>}>
          <SignInForm />
        </Suspense>
      </main>

      <Footer />
    </div>
  );
}
