"use client";
import React, { useState } from "react";
import Link from "next/link";
import { useSession, signIn, signOut } from "next-auth/react";

export default function UserMenu() {
  const { data: session, status } = useSession();
  const [open, setOpen] = useState(false);

  if (status === "loading") {
    return <div className="h-8 w-20 bg-[#1A1A1A] animate-pulse rounded-md"></div>;
  }

  if (!session || !session.user) {
    return (
      <button
        onClick={() => signIn()}
        className="px-4 py-1.5 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] text-sm font-semibold rounded-md transition cursor-pointer"
      >
        Sign In
      </button>
    );
  }

  const user = session.user;
  const tier = (user as any).tier || "FREE";

  const tierColors: Record<string, string> = {
    FREE: "bg-slate-800 text-slate-300 border-slate-700",
    BASIC: "bg-[#34D399]/10 text-[#34D399] border-[#34D399]/30",
    PRO: "bg-purple-950/40 text-purple-400 border-purple-500/30",
  };

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 p-1 rounded-full border border-[#222222] bg-[#111111] hover:border-[#333333] transition cursor-pointer"
      >
        {user.image ? (
          <img src={user.image} alt={user.name || "User Avatar"} className="w-8 h-8 rounded-full" />
        ) : (
          <div className="w-8 h-8 rounded-full bg-[#222222] flex items-center justify-center text-xs font-bold text-[#F3F3F3]">
            {user.name ? user.name[0].toUpperCase() : "U"}
          </div>
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-[#111111] border border-[#222222] rounded-xl shadow-2xl p-2 z-50 space-y-1">
          <div className="px-3 py-2 border-b border-[#222222]">
            <p className="text-sm font-bold text-[#F3F3F3] truncate">{user.name || "User Account"}</p>
            <p className="text-xs text-[#8E8E8E] truncate mb-2">{user.email}</p>
            <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${tierColors[tier] || tierColors.FREE}`}>
              {tier} PLAN
            </span>
          </div>

          <Link
            href="/account"
            onClick={() => setOpen(false)}
            className="block px-3 py-2 text-xs text-[#CCCCCC] hover:text-white hover:bg-[#1A1A1A] rounded-md transition"
          >
            ⚙️ Account Settings
          </Link>
          <Link
            href="/pricing"
            onClick={() => setOpen(false)}
            className="block px-3 py-2 text-xs text-[#34D399] hover:bg-[#1A1A1A] rounded-md transition"
          >
            ⚡ Upgrade Plan
          </Link>

          <button
            onClick={() => signOut()}
            className="w-full text-left px-3 py-2 text-xs text-red-400 hover:bg-red-500/10 rounded-md transition cursor-pointer"
          >
            🚪 Sign Out
          </button>
        </div>
      )}
    </div>
  );
}
