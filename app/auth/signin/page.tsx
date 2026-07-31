"use client";
import React, { useState, useEffect, Suspense } from "react";
import { signIn } from "next-auth/react";
import { useRouter, useSearchParams } from "next/navigation";
import { Navbar } from "@/app/components/Navbar";
import { Footer } from "@/app/components/Footer";

function SignInForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlError = searchParams?.get("error");

  const [tab, setTab] = useState<"signin" | "register">("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (urlError) {
      if (urlError === "OAuthSignin") {
        setError("ระบบ Social Login กำลังรอใส่ Client ID / Secret ของผู้ให้บริการ");
      } else if (urlError === "CredentialsSignin") {
        setError("อีเมลหรือรหัสผ่านไม่ถูกต้อง");
      } else {
        setError("เกิดข้อผิดพลาดในการเข้าสู่ระบบ");
      }
    }
  }, [urlError]);

  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (tab === "register") {
      try {
        const res = await fetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, email, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "การลงทะเบียนไม่สำเร็จ");

        // Automatically sign in after registration
        const result = await signIn("credentials", { email, password, redirect: false });
        if (result?.error) throw new Error(result.error);
        router.push("/");
      } catch (err: any) {
        setError(err.message || "เกิดข้อผิดพลาดในการสมัครสมาชิก");
      } finally {
        setLoading(false);
      }
    } else {
      const result = await signIn("credentials", { email, password, redirect: false });
      setLoading(false);
      if (result?.error) {
        setError("อีเมลหรือรหัสผ่านไม่ถูกต้อง");
      } else {
        router.push("/");
      }
    }
  };

  return (
    <div className="bg-[#111111] border border-[#222222] rounded-2xl p-8 max-w-md w-full shadow-2xl space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Welcome to HarmoniQ</h1>
        <p className="text-xs text-[#8E8E8E]">AI Music Separator & Audio Toolkit</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-[#222222]">
        <button
          onClick={() => { setTab("signin"); setError(null); }}
          className={`flex-1 py-2 text-sm font-semibold transition border-b-2 cursor-pointer ${
            tab === "signin"
              ? "border-[#F97316] text-[#F97316]"
              : "border-transparent text-[#8E8E8E] hover:text-[#F5F0EB]"
          }`}
        >
          Sign In
        </button>
        <button
          onClick={() => { setTab("register"); setError(null); }}
          className={`flex-1 py-2 text-sm font-semibold transition border-b-2 cursor-pointer ${
            tab === "register"
              ? "border-[#F97316] text-[#F97316]"
              : "border-transparent text-[#8E8E8E] hover:text-[#F5F0EB]"
          }`}
        >
          Create Account
        </button>
      </div>

      {error && <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg">{error}</div>}

      {/* Email & Password Form */}
      <form onSubmit={handleCredentialsSubmit} className="space-y-4">
        {tab === "register" && (
          <div>
            <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Full Name</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="John Doe"
              className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#F97316]"
            />
          </div>
        )}
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Email Address</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@example.com"
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#F97316]"
          />
        </div>
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Password</label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#F97316]"
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-bold text-sm rounded-lg transition cursor-pointer"
        >
          {loading ? "Processing..." : tab === "signin" ? "Sign In with Email" : "Create Account"}
        </button>
      </form>

      <div className="relative flex items-center justify-center border-t border-[#222222] pt-4">
        <span className="bg-[#111111] px-2 text-[10px] uppercase text-[#8E8E8E] font-bold">Or continue with</span>
      </div>

      {/* Social Logins */}
      <div className="space-y-2">
        <button
          onClick={() => signIn("google")}
          className="w-full py-2 bg-[#1E1B18] hover:bg-[#222222] border border-[#36322E] text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
        >
          🌐 Google
        </button>
        <button
          onClick={() => signIn("facebook")}
          className="w-full py-2 bg-[#1877F2]/10 hover:bg-[#1877F2]/20 border border-[#1877F2]/40 text-[#1877F2] text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
        >
          📘 Facebook
        </button>
        <button
          onClick={() => signIn("line")}
          className="w-full py-2 bg-[#00C300]/10 hover:bg-[#00C300]/20 border border-[#00C300]/40 text-[#00C300] text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
        >
          💬 LINE Login
        </button>
      </div>
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
