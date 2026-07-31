"use client";

import React, { Suspense, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { signIn, useSession } from "next-auth/react";
import { toast } from "sonner";
import { AlertTriangle, ShieldCheck } from "lucide-react";
import { Navbar } from "../../components/Navbar";
import { Footer } from "../../components/Footer";
import { OAUTH_PROVIDERS } from "@/lib/config";

// M19+: หน้าให้ OAuth-only user ยืนยันตัวตนผ่าน provider อีกครั้ง
// ก่อนดำเนินการที่ทำลายข้อมูล (ลบบัญชี / ยกเลิก subscription)
const REAUTH_WINDOW_MS = 5 * 60 * 1000;

function ConfirmDeleteContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const action = searchParams.get("action") === "cancel" ? "cancel" : "delete";
  const done = searchParams.get("done");

  const { data: session, status } = useSession();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isOAuthOnly = !!session?.user && !(session.user as { hasPassword?: boolean }).hasPassword;

  // กลับมาหลัง re-auth (done=1) -> session มี reauthAt ใหม่ -> เรียก API อัตโนมัติ
  useEffect(() => {
    if (done !== "1" || status !== "authenticated" || !session?.user) return;

    const reauthAt = (session.user as { reauthAt?: number }).reauthAt;
    if (!reauthAt || Date.now() - reauthAt > REAUTH_WINDOW_MS) {
      setError("การยืนยันหมดอายุแล้ว กรุณากดยืนยันอีกครั้ง");
      return;
    }

    setSubmitting(true);
    const apiPath = action === "cancel" ? "/api/subscription/cancel" : "/api/account";
    const method = action === "cancel" ? "POST" : "DELETE";

    fetch(apiPath, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data.error || "ดำเนินการไม่สำเร็จ");
        }
        toast.success(action === "cancel" ? "ยกเลิกสมาชิกเรียบร้อย" : "ลบบัญชีเรียบร้อย");
        router.push(action === "cancel" ? "/account" : "/");
      })
      .catch((err: Error) => {
        setError(err.message || "ดำเนินการไม่สำเร็จ กรุณาลองใหม่");
      })
      .finally(() => setSubmitting(false));
  }, [done, status, session, action, router]);

  if (status === "loading") {
    return <div className="py-24 text-center text-[#8E8E8E] text-sm">Loading...</div>;
  }

  if (!session?.user) {
    return (
      <div className="py-24 text-center space-y-4 px-4">
        <h2 className="text-xl font-bold">กรุณาเข้าสู่ระบบก่อน</h2>
        <button
          onClick={() => router.push("/auth/signin")}
          className="px-6 py-3 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-semibold text-xs rounded-xl transition"
        >
          เข้าสู่ระบบ
        </button>
      </div>
    );
  }

  if (!isOAuthOnly) {
    // ผู้ใช้ที่มี password ใช้ flow ยืนยันด้วยรหัสผ่านที่หน้า /account ตามเดิม
    router.replace("/account");
    return null;
  }

  return (
    <div className="max-w-md mx-auto py-16 px-4 text-center space-y-6">
      <div className="w-14 h-14 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center mx-auto text-red-400">
        <AlertTriangle className="w-7 h-7" />
      </div>
      <div>
        <h2 className="text-xl font-bold text-white">
          {action === "cancel" ? "ยืนยันการยกเลิกสมาชิก" : "ยืนยันการลบบัญชี"}
        </h2>
        <p className="text-xs text-[#8E8E8E] mt-2 leading-relaxed">
          เพื่อความปลอดภัย กรุณายืนยันตัวตนผ่านบัญชีที่เชื่อมต่ออีกครั้ง
          (การดำเนินการนี้ทำไม่ได้ถ้าไม่ได้ยืนยัน)
        </p>
      </div>

      {error && (
        <div className="rounded-lg bg-red-950/40 border border-red-500/30 px-3 py-2.5 text-xs text-red-400">
          {error}
        </div>
      )}

      <div className="space-y-2">
        {OAUTH_PROVIDERS.map((provider) => (
          <button
            key={provider.id}
            onClick={() =>
              signIn(provider.id, {
                callbackUrl: `/account/confirm-delete?action=${action}&done=1`,
              })
            }
            disabled={submitting}
            className="w-full flex items-center justify-center gap-2 rounded-xl border border-[#2C2824] bg-[#161412] px-4 py-3 text-sm font-semibold text-white hover:border-[#E5A93D]/50 hover:text-[#E5A93D] transition disabled:opacity-50"
          >
            <ShieldCheck className="w-4 h-4" />
            ยืนยันด้วย {provider.name}
          </button>
        ))}
      </div>

      <button
        onClick={() => router.back()}
        className="text-xs text-[#666] hover:text-white transition"
      >
        กลับไปหน้าก่อนหน้า
      </button>
    </div>
  );
}

export default function ConfirmDeletePage() {
  return (
    <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col">
      <Navbar />
      <main className="flex-grow">
        <Suspense fallback={<div className="py-24 text-center text-[#8E8E8E] text-sm">Loading...</div>}>
          <ConfirmDeleteContent />
        </Suspense>
      </main>
      <Footer />
    </div>
  );
}
