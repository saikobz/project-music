"use client";
import React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-[#5B21B6]/10 bg-[#0B1021]/85 py-6 text-center text-xs text-[#A78BFA]/60">
      <div className="mx-auto max-w-6xl px-4 space-y-1.5">
        <p>© 2026 HarmoniQ. All rights reserved.</p>
        <p className="text-[11px] text-[#A78BFA]/40">
          ระบบแยกสเตมเพลงและปรับระดับเสียงอัตโนมัติด้วยปัญญาประดิษฐ์
        </p>
      </div>
    </footer>
  );
};
