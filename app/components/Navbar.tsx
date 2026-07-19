"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

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
  // สถานะเปิด/ปิดเมนูสำหรับมือถือ
  const [mobileOpen, setMobileOpen] = useState(false);

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

        {/* Hamburger */}
        <div className="flex items-center gap-3">
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
