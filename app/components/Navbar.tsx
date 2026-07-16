"use client";
import React, { useEffect, useState } from "react";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export const Navbar: React.FC = () => {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE}/health`, { timeout: 3000 });
        if (res.data?.status === "ok") {
          setApiOnline(true);
        } else {
          setApiOnline(false);
        }
      } catch (err) {
        setApiOnline(false);
      }
    };

    // เช็กสถานะทันทีที่โหลดหน้าเว็บ
    checkStatus();

    // เช็กสถานะเป็นระยะทุก ๆ 10 วินาที
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[#2A2A2A] bg-[#0A0A0A]/80 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        {/* โลโก้ของระบบ */}
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-[#F3F3F3]">
            HarmoniQ
          </span>
          <span className="hidden text-xs font-medium text-[#8E8E8E] sm:inline">
            AI Audio Toolkit
          </span>
        </div>

        {/* เมนูเชื่อมต่อ API Status Check */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 rounded-full bg-[#121212] px-3 py-1 border border-[#2A2A2A] text-xs font-medium">
            <span
              className={`h-2.5 w-2.5 rounded-full transition-all duration-500 ${
                apiOnline === null
                  ? "bg-amber-500 shadow-[0_0_8px_#f59e0b]"
                  : apiOnline
                  ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
                  : "bg-rose-500 shadow-[0_0_8px_#f43f5e]"
              }`}
            />
            <span className="text-[#8E8E8E]">
              API Status:{" "}
              {apiOnline === null
                ? "Checking..."
                : apiOnline
                ? "Online"
                : "Offline"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
