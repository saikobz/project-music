"use client";

import React, { useEffect } from "react";

type ErrorProps = {
  error: Error & { digest?: string };
  reset: () => void;
};

export default function ErrorBoundary({ error, reset }: ErrorProps) {
  useEffect(() => {
    // บันทึก Log ข้อผิดพลาดของระบบลง Console
    console.error("Unhandled runtime error captured:", error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-[#050505] p-6 text-[#F3F3F3]">
      <div className="w-full max-w-md rounded-2xl border border-[#2A2A2A] bg-[#0A0A0A] p-8 text-center shadow-2xl backdrop-blur-md">
        {/* Warning Icon with premium glowing effect */}
        <div className="mx-auto mb-6 flex h-16 w-16 items-center justify-center rounded-full bg-[#E5A93D]/10 border border-[#E5A93D]/30 text-[#E5A93D] shadow-[0_0_20px_rgba(229,169,61,0.15)]">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-8 w-8"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>

        <h2 className="mb-3 text-2xl font-bold tracking-tight text-white">
          เกิดข้อผิดพลาดในการโหลดหน้าเว็บ
        </h2>
        
        <p className="mb-8 text-sm text-[#8E8E8E] leading-relaxed">
          ระบบพบปัญหาในการเรนเดอร์ส่วนติดต่อผู้ใช้งานหรือข้อมูลบางอย่างขัดข้อง กรุณาลองกดปุ่มด้านล่างเพื่อลองใหม่อีกครั้ง
        </p>

        <div className="flex flex-col gap-3">
          <button
            onClick={() => reset()}
            className="w-full rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0A0A0A] transition-all hover:shadow-[0_0_25px_rgba(229,169,61,0.35)] cursor-pointer"
          >
            ลองใหม่อีกครั้ง (Try Again)
          </button>
          
          <button
            onClick={() => window.location.reload()}
            className="w-full rounded-xl border border-[#2A2A2A] bg-[#121212] px-4 py-3.5 font-semibold text-[#8E8E8E] hover:text-white hover:border-[#444444] transition-colors cursor-pointer"
          >
            รีโหลดทั้งหน้า (Force Reload)
          </button>
        </div>

        {error.digest && (
          <div className="mt-6 text-[10px] font-mono text-[#333333]">
            Error Digest: {error.digest}
          </div>
        )}
      </div>
    </div>
  );
}
