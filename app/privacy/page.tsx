"use client";

import React from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import { Lock, ShieldCheck, Database, Trash2 } from "lucide-react";

export default function PrivacyPage() {
  return (
    <div className="flex min-h-screen flex-col bg-[#0D0B0A] text-[#F5F0EB]">
      <Navbar />

      <main className="flex-grow max-w-4xl mx-auto px-4 py-12 md:py-16 space-y-8">
        <header className="space-y-3 border-b border-[#2C2824] pb-6">
          <div className="inline-flex items-center gap-2 text-[#F97316] text-xs font-semibold uppercase tracking-wider">
            <Lock className="w-4 h-4" />
            <span>Data Protection & Privacy Policy</span>
          </div>
          <h1 className="text-3xl md:text-5xl font-bold">นโยบายความเป็นส่วนตัว (Privacy Policy)</h1>
          <p className="text-sm text-[#8E8E8E]">ปรับปรุงล่าสุดเมื่อ: 25 กรกฎาคม 2026</p>
        </header>

        <article className="space-y-8 text-sm md:text-base text-[#A09890] leading-relaxed">
          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              1. การคุ้มครองข้อมูลส่วนบุคคล (PDPA Compliance)
            </h2>
            <p>
              HarmoniQ ให้ความสำคัญกับความเป็นส่วนตัวของผู้ใช้ ข้อมูลบัญชีผู้ใช้ (ชื่อ, อีเมล, รูปโปรไฟล์) ที่ได้รับจากระบบ NextAuth (Google/Email) จะถูกจัดเก็บด้วยความปลอดภัยสูงเพื่อใช้ระบุตัวตนและจัดการโควตาสมาชิกเท่านั้น และจะไม่ถูกนำไปขายหรือเปิดเผยแก่บุคคลภายนอก
            </p>
          </section>

          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <Trash2 className="w-5 h-5 text-[#F97316]" />
              2. นโยบายลบไฟล์เสียงชั่วคราว (Temporary Audio Retention & Cleanup)
            </h2>
            <p>
              ไฟล์เพลง WAV ที่ผู้ใช้อัปโหลดเพื่อแยกสเต็ม จะถูกนำไปประมวลผลบน AI Server ในสภาพแวดล้อมชั่วคราว (Temporary Sandbox) ไฟล์เสียงทั้งหมดจะถูกลบออกจากดิสก์ของเซิร์ฟเวอร์โดยอัตโนมัติภายใน 24 ชั่วโมงหลังจากการประมวลผลเสร็จสิ้นเพื่อความปลอดภัย
            </p>
          </section>

          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-[#F97316]" />
              3. ระบบชำระเงินและคุกกี้ (Payments & Cookies)
            </h2>
            <p>
              ข้อมูลบัตรเครดิตและระบบ PromptPay ทั้งหมดถูกจัดการอย่างปลอดภัยโดยตรงผ่าน Payment Gateway มาตรฐาน **Omise (Opn Payments)** HarmoniQ ไม่มีนโยบายจัดเก็บเลขบัตรเครดิตหรือรหัส CVV ของผู้ใช้งานไว้ในฐานข้อมูลของเราเอง
            </p>
          </section>
        </article>
      </main>

      <Footer />
    </div>
  );
}
