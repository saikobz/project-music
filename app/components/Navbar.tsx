"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

// รายการเมนูหลักของ Navbar
const NAV_LINKS = [
  { href: "/", label: "Workspace" },
  { href: "/models", label: "AI Models" },
  { href: "/guide", label: "Guide" },
  { href: "/api-pricing", label: "API & Pricing" },
  { href: "/about", label: "About" },
] as const;

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  // สถานะเปิด/ปิดเมนูสำหรับมือถือ
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await axios.get(`${API_BASE}/health`, { timeout: 3000 });
        setApiOnline(res.data?.status === "ok");
      } catch {
        setApiOnline(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  // ปิดเมนูมือถือเมื่อเปลี่ยนหน้า
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[#1E1E1E] bg-[#0A0A0A]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        {/* โลโก้ */}
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <span className="text-lg font-bold tracking-tight text-[#F3F3F3]">HarmoniQ</span>
          <span className="hidden text-xs font-medium text-[#555555] sm:inline">AI Audio Toolkit</span>
        </Link>

        {/* เมนู Desktop */}
        <nav className="hidden md:flex items-center gap-1">
          {NAV_LINKS.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? "bg-[#1A1A1A] text-[#F3F3F3]"
                    : "text-[#555555] hover:text-[#BBBBBB] hover:bg-[#111111]"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        {/* API Status + Hamburger */}
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex items-center gap-1.5 rounded-full bg-[#111111] px-3 py-1 border border-[#1E1E1E] text-xs font-medium">
            <span className={`h-2 w-2 rounded-full transition-all duration-500 ${
              apiOnline === null ? "bg-amber-500 shadow-[0_0_6px_#f59e0b]"
              : apiOnline ? "bg-emerald-500 shadow-[0_0_6px_#10b981]"
              : "bg-rose-500 shadow-[0_0_6px_#f43f5e]"
            }`} />
            <span className="text-[#555555]">{apiOnline === null ? "Checking…" : apiOnline ? "Online" : "Offline"}</span>
          </div>

          {/* ปุ่ม Hamburger สำหรับมือถือ */}
          <button
            className="md:hidden p-2 rounded-md text-[#555555] hover:text-[#F3F3F3] hover:bg-[#111111] transition cursor-pointer"
            onClick={() => setMobileOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            {mobileOpen ? (
              <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-5 h-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile Dropdown Menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-[#1E1E1E] bg-[#0A0A0A] px-4 py-3 space-y-1">
          {NAV_LINKS.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`block px-3 py-2 rounded-md text-sm font-medium transition ${
                  isActive ? "bg-[#1A1A1A] text-[#F3F3F3]" : "text-[#555555] hover:text-[#F3F3F3] hover:bg-[#111111]"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
};
