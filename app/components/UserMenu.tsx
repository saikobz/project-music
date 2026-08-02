"use client";
import React, { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useSession, signIn, signOut } from "next-auth/react";
import { Settings, Crown, LogOut, LayoutDashboard, History, ChevronDown, User, Check } from "lucide-react";

export default function UserMenu() {
  const { data: session, status } = useSession();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // ปิดเมนูเมื่อคลิกข้างนอก dropdown
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    if (open) document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  if (status === "loading") {
    return <div className="h-8 w-20 bg-[#1E1B18] animate-pulse rounded-md"></div>;
  }

  if (!session || !session.user) {
    return (
      <button
        onClick={() => signIn()}
        className="px-4 py-1.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white text-sm font-semibold rounded-md transition cursor-pointer"
      >
        Sign In
      </button>
    );
  }

  const user = session.user;
  const tier = (user as any).tier || "FREE";

  // ลำดับสี Tier: Free (เทา) → Basic (ทอง) → Pro (Coral) — สื่อระดับที่สูงขึ้น
  const tierColors: Record<string, string> = {
    FREE: "bg-slate-800 text-slate-300 border-slate-700",
    BASIC: "bg-[#E5A93D]/10 text-[#E5A93D] border-[#E5A93D]/30",
    PRO: "bg-[#F97316]/10 text-[#F97316] border-[#F97316]/20",
  };

  const close = () => setOpen(false);

  return (
    <div className="relative" ref={menuRef}>
      {/* Trigger button */}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="User menu"
        className={`flex items-center gap-1.5 p-1 pl-1 rounded-full border transition cursor-pointer ${
          open
            ? "border-[#F97316]/40 bg-[#1E1B18]"
            : "border-[#2C2824] bg-[#161412] hover:border-[#36322E]"
        }`}
      >
        {user.image ? (
          <img src={user.image} alt={user.name || "User Avatar"} className="w-8 h-8 rounded-full" />
        ) : (
          <div className="w-8 h-8 rounded-full bg-[#2C2824] flex items-center justify-center text-xs font-bold text-[#F5F0EB]">
            {user.name ? user.name[0].toUpperCase() : "U"}
          </div>
        )}
        <ChevronDown
          className={`w-4 h-4 text-[#8E8E8E] transition-transform duration-200 ${open ? "rotate-180" : ""}`}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div className="absolute right-0 mt-2 w-64 bg-[#161412] border border-[#2C2824] rounded-2xl shadow-xl ring-1 ring-black/50 p-1.5 z-50 animate-fade-in-up origin-top-right">
          {/* Header: ข้อมูลผู้ใช้ */}
          <div className="px-3 py-3 border-b border-[#2C2824] mb-1">
            <div className="flex items-center gap-3">
              {user.image ? (
                <img src={user.image} alt={user.name || "User Avatar"} className="w-10 h-10 rounded-full" />
              ) : (
                <div className="w-10 h-10 rounded-full bg-[#2C2824] flex items-center justify-center text-sm font-bold text-[#F5F0EB]">
                  {user.name ? user.name[0].toUpperCase() : "U"}
                </div>
              )}
              <div className="min-w-0 flex-1">
                <p className="text-sm font-bold text-[#F5F0EB] truncate">{user.name || "User Account"}</p>
                <p className="text-[11px] text-[#8E8E8E] truncate">{user.email}</p>
              </div>
            </div>
            <span
              className={`mt-2 inline-flex items-center gap-1 text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                tierColors[tier] || tierColors.FREE
              }`}
            >
              {tier === "PRO" ? (
                <Crown className="w-3 h-3" />
              ) : tier === "BASIC" ? (
                <Check className="w-3 h-3" />
              ) : (
                <User className="w-3 h-3" />
              )}
              {tier} PLAN
            </span>
          </div>

          {/* กลุ่มงานหลัก */}
          <Link
            href="/studio"
            onClick={close}
            className="flex items-center gap-2.5 px-3 py-2 text-xs text-[#A09890] hover:text-white hover:bg-[#1E1B18] rounded-lg transition-colors"
          >
            <LayoutDashboard className="w-4 h-4 text-[#5C5854]" />
            Studio Dashboard
          </Link>
          <Link
            href="/dashboard/history"
            onClick={close}
            className="flex items-center gap-2.5 px-3 py-2 text-xs text-[#A09890] hover:text-white hover:bg-[#1E1B18] rounded-lg transition-colors"
          >
            <History className="w-4 h-4 text-[#5C5854]" />
            Processing History
          </Link>

          {/* กลุ่มบัญชี */}
          <div className="border-t border-[#2C2824] my-1"></div>
          <Link
            href="/account"
            onClick={close}
            className="flex items-center gap-2.5 px-3 py-2 text-xs text-[#A09890] hover:text-white hover:bg-[#1E1B18] rounded-lg transition-colors"
          >
            <Settings className="w-4 h-4 text-[#5C5854]" />
            Account Settings
          </Link>
          <Link
            href="/pricing"
            onClick={close}
            className="flex items-center gap-2.5 px-3 py-2 text-xs text-[#F97316] font-semibold hover:bg-[#1E1B18] rounded-lg transition-colors"
          >
            <Crown className="w-4 h-4" />
            Upgrade Plan
          </Link>

          {/* ออกจากระบบ */}
          <div className="border-t border-[#2C2824] my-1"></div>
          <button
            onClick={() => signOut()}
            className="w-full flex items-center gap-2.5 px-3 py-2 text-xs text-[#EF4444] hover:bg-[#EF4444]/10 rounded-lg transition-colors cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
