"use client";
import React, { useState, Suspense } from "react";
import Link from "next/link";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Navbar } from "@/app/components/Navbar";
import { Footer } from "@/app/components/Footer";
import PasswordInput from "@/app/components/PasswordInput";
import SocialButtons from "@/app/components/SocialButtons";

type Strength = { level: "weak" | "medium" | "strong"; color: string; label: string };

// คำนวณความแข็งแรงของรหัสผ่านแบบ real-time
// - Weak: สั้นกว่า 8 หรือมีประเภทตัวอักษรน้อย
// - Medium: ≥8 ตัว + ผสมตัวเลข/ตัวพิมพ์ใหญ่
// - Strong: ≥8 ตัว + ตัวพิมพ์ใหญ่ + ตัวเลข + อักขระพิเศษ
function getStrength(pwd: string): Strength {
  if (!pwd) return { level: "weak", color: "#5C5854", label: "" };
  const hasUpper = /[A-Z]/.test(pwd);
  const hasDigit = /\d/.test(pwd);
  const hasSpecial = /[^A-Za-z0-9]/.test(pwd);

  if (pwd.length >= 8 && hasUpper && hasDigit && hasSpecial) {
    return { level: "strong", color: "#34D399", label: "Strong" };
  }
  if (pwd.length >= 8 && (hasUpper || hasDigit)) {
    return { level: "medium", color: "#F59E0B", label: "Medium" };
  }
  return { level: "weak", color: "#EF4444", label: "Weak" };
}

function RegisterForm() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const strength = getStrength(password);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Client-side validation ก่อนส่ง API
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError("รูปแบบอีเมลไม่ถูกต้อง");
      return;
    }
    if (password.length < 6) {
      setError("รหัสผ่านต้องมีความยาวอย่างน้อย 6 ตัวอักษร");
      return;
    }
    if (password !== confirmPassword) {
      setError("รหัสผ่านไม่ตรงกัน");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "การลงทะเบียนไม่สำเร็จ");

      // สมัครสำเร็จ -> เข้าสู่ระบบอัตโนมัติ
      const result = await signIn("credentials", { email, password, redirect: false });
      if (result?.error) throw new Error(result.error);
      router.push("/");
    } catch (err: any) {
      setError(err.message || "เกิดข้อผิดพลาดในการสมัครสมาชิก");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#161412] border border-[#2C2824] rounded-2xl p-8 max-w-md w-full shadow-2xl space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">Create Account</h1>
        <p className="text-xs text-[#8E8E8E]">Join HarmoniQ — AI Music Separator & Audio Toolkit</p>
      </div>

      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-xs rounded-lg">{error}</div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Full Name</label>
          <input
            type="text"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="John Doe"
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 text-sm text-white focus:outline-none focus:border-[#F97316] transition"
          />
        </div>
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
          {password && (
            <div className="mt-2">
              <div className="flex gap-1">
                <div
                  className={`h-1 flex-1 rounded-full transition-colors ${
                    strength.level === "weak" ? "bg-[#EF4444]" : "bg-[#36322E]"
                  }`}
                />
                <div
                  className={`h-1 flex-1 rounded-full transition-colors ${
                    strength.level === "medium"
                      ? "bg-[#F59E0B]"
                      : strength.level === "strong"
                        ? "bg-[#34D399]"
                        : "bg-[#36322E]"
                  }`}
                />
                <div
                  className={`h-1 flex-1 rounded-full transition-colors ${
                    strength.level === "strong" ? "bg-[#34D399]" : "bg-[#36322E]"
                  }`}
                />
              </div>
              <span className="text-[10px] font-medium mt-1 inline-block" style={{ color: strength.color }}>
                {strength.label}
              </span>
            </div>
          )}
        </div>
        <div>
          <label className="block text-xs font-semibold text-[#8E8E8E] mb-1">Confirm Password</label>
          <PasswordInput value={confirmPassword} onChange={setConfirmPassword} />
          {confirmPassword && password !== confirmPassword && (
            <p className="text-[10px] text-[#EF4444] mt-1">รหัสผ่านไม่ตรงกัน</p>
          )}
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-bold text-sm rounded-lg transition cursor-pointer"
        >
          {loading ? "Processing..." : "Create Account"}
        </button>
      </form>

      <div className="relative flex items-center justify-center border-t border-[#2C2824] pt-4">
        <span className="bg-[#161412] px-2 text-[10px] uppercase text-[#8E8E8E] font-bold">Or continue with</span>
      </div>

      <SocialButtons />

      <p className="text-center text-xs text-[#8E8E8E]">
        มีบัญชีอยู่แล้ว?{" "}
        <Link href="/auth/signin" className="text-[#F97316] font-semibold hover:underline">
          Sign In
        </Link>
      </p>
    </div>
  );
}

export default function SignupPage() {
  return (
    <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow flex items-center justify-center p-4 py-12">
        <Suspense fallback={<div className="text-sm text-[#8E8E8E]">Loading...</div>}>
          <RegisterForm />
        </Suspense>
      </main>

      <Footer />
    </div>
  );
}
