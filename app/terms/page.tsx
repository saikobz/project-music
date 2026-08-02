"use client";

import React from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import { FileText, ShieldAlert, Scale, UserCheck } from "lucide-react";

export default function TermsPage() {
  return (
    <div className="flex min-h-screen flex-col bg-[#0D0B0A] text-[#F5F0EB]">
      <Navbar />

      <main className="flex-grow max-w-4xl mx-auto px-4 py-12 md:py-16 space-y-8">
        <header className="space-y-3 border-b border-[#2C2824] pb-6">
          <div className="inline-flex items-center gap-2 text-[#F97316] text-xs font-semibold uppercase tracking-wider">
            <FileText className="w-4 h-4" />
            <span>HarmoniQ Legal Compliance</span>
          </div>
          <h1 className="text-3xl md:text-5xl font-bold">เงื่อนไขและข้อตกลงการใช้งาน (Terms of Service)</h1>
          <p className="text-sm text-[#8E8E8E]">ปรับปรุงล่าสุดเมื่อ: 25 กรกฎาคม 2026</p>
        </header>

        <article className="space-y-8 text-sm md:text-base text-[#A09890] leading-relaxed">
          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <UserCheck className="w-5 h-5 text-[#F97316]" />
              1. สิทธิ์และการรับรองในไฟล์เสียง (Audio Copyright Ownership)
            </h2>
            <p>
              ผู้ใช้งานเป็นผู้รับผิดชอบต่อไฟล์เสียงทั้งหมดที่อัปโหลดเข้าสู่ระบบ HarmoniQ โดยต้องเป็นเจ้าของลิขสิทธิ์อย่างถูกต้อง หรือได้รับใบอนุญาต (License) ในการดัดแปลงและแยกแทร็กเสียง ห้ามใช้นำไฟล์เสียงที่ละเมิดลิขสิทธิ์มาประมวลผลเพื่อการค้าโดยไม่ได้รับอนุญาต
            </p>
          </section>

          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <Scale className="w-5 h-5 text-[#F97316]" />
              2. ข้อจำกัดและโควตาการใช้งาน (Service Quotas & Restrictions)
            </h2>
            <p>
              การใช้งาน HarmoniQ ถูกแบ่งตามแพ็กเกจ (Free, Basic, Pro) ซึ่งมีโควตาจำนวนเพลงต่อเดือน และข้อจำกัดฟีเจอร์แตกต่างกันออกไป ห้ามมิให้ผู้ใช้งานพยายามย้อนรอยโค้ด (Reverse Engineer) หรือส่งคำขออัปโหลดในปริมาณมากผิดปกติผ่านบอทเพื่อก่อกวนระบบ (DDoS)
            </p>
          </section>

          <section className="space-y-3 bg-[#161412] p-6 rounded-2xl border border-[#2C2824]">
            <h2 className="text-xl font-semibold text-white flex items-center gap-2">
              <ShieldAlert className="w-5 h-5 text-[#F97316]" />
              3. นโยบายการคืนเงินและการยกเลิก (Refund & Cancellation)
            </h2>
            <p>
              ผู้ใช้งานสามารถยกเลิกการต่ออายุสมาชิกแบบรายเดือนได้ตลอดเวลาผ่านหน้า `/account` ทั้งนี้ ค่าบริการ subscription ที่ถูกตัดไปแล้วสำหรับรอบปัจจุบันจะไม่สามารถขอคืนเงินได้ เว้นแต่เกิดข้อผิดพลาดจากระบบตัดเงิน Omise หรือระบบไม่ส่งมอบบริการตามปกติ
            </p>
          </section>
        </article>
      </main>

      <Footer />
    </div>
  );
}
