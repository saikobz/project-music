"use client";

import React from "react";
import Link from "next/link";
import { Music } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-[#1E1E1E] bg-[#080808] py-12 mt-auto text-[#888888]">
      <div className="mx-auto max-w-6xl px-4 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="space-y-3 md:col-span-1">
            <div className="flex items-center gap-2 text-white font-bold text-lg">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-purple-600 to-indigo-600 flex items-center justify-center">
                <Music className="w-4 h-4 text-white" />
              </div>
              <span>HarmoniQ</span>
            </div>
            <p className="text-xs text-[#777777] leading-relaxed max-w-xs">
              ระบบ AI แยกแทร็กเสียงดนตรีและมาสเตอริ่งสำหรับโปรดิวเซอร์มืออาชีพ
            </p>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">ผลิตภัณฑ์</h4>
            <ul className="space-y-2 text-xs">
              <li><Link href="/studio" className="hover:text-white transition-colors">Studio Workspace</Link></li>
              <li><Link href="/pricing" className="hover:text-white transition-colors">ราคาและแพ็กเกจ</Link></li>
              <li><Link href="/models" className="hover:text-white transition-colors">โมเดล AI (Open-Unmix)</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">ทรัพยากร &amp; กฎหมาย</h4>
            <ul className="space-y-2 text-xs">
              <li><Link href="/guide" className="hover:text-white transition-colors">คู่มือการใช้งาน</Link></li>
              <li><Link href="/terms" className="hover:text-white transition-colors">เงื่อนไขการใช้งาน (Terms)</Link></li>
              <li><Link href="/privacy" className="hover:text-white transition-colors">นโยบายความเป็นส่วนตัว (Privacy)</Link></li>
            </ul>
          </div>

          <div>
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-3">ช่วยเหลือ</h4>
            <ul className="space-y-2 text-xs">
              <li><Link href="/support" className="hover:text-white transition-colors">ศูนย์ช่วยเหลือ &amp; ติดต่อ</Link></li>
              <li><Link href="/about" className="hover:text-white transition-colors">เกี่ยวกับ HarmoniQ</Link></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-[#181818] pt-6 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#555555]">
          <p>© {new Date().getFullYear()} HarmoniQ Inc. All rights reserved.</p>
          <div className="flex items-center gap-2 text-[#666666]">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>All Systems Operational</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
