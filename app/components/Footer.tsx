"use client";
import React from "react";

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-[#2A2A2A] bg-[#0A0A0A] py-6 text-center text-xs text-[#666666]">
      <div className="mx-auto max-w-6xl px-4 space-y-1.5">
        <p>© 2026 HarmoniQ. All rights reserved.</p>
        <p className="text-[11px] text-[#555555]">
          ระบบแยกสเตมเพลงและปรับระดับเสียงอัตโนมัติด้วยปัญญาประดิษฐ์
        </p>
      </div>
    </footer>
  );
};
