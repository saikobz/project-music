"use client";

import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import { HelpCircle, Mail, MessageSquare, Send, CheckCircle2 } from "lucide-react";

export default function SupportPage() {
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#0D0B0A] text-[#F5F0EB]">
      <Navbar />

      <main className="flex-grow max-w-4xl mx-auto px-4 py-12 md:py-16 space-y-8">
        <header className="space-y-3 border-b border-[#222] pb-6">
          <div className="inline-flex items-center gap-2 text-indigo-400 text-xs font-semibold uppercase tracking-wider">
            <HelpCircle className="w-4 h-4" />
            <span>Help &amp; Customer Support</span>
          </div>
          <h1 className="text-3xl md:text-5xl font-bold">ศูนย์ช่วยเหลือและแจ้งปัญหา (Support Center)</h1>
          <p className="text-sm text-[#8E8E8E]">ทีมงานพร้อมดูแลและตอบคำถามข้อสงสัยเกี่ยวกับระบบ HarmoniQ</p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1 space-y-4">
            <div className="p-6 rounded-2xl bg-[#111111] border border-[#222] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                <Mail className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-white">อีเมลช่วยเหลือ</h3>
              <p className="text-xs text-[#8E8E8E]">support@harmoniq.ai</p>
            </div>

            <div className="p-6 rounded-2xl bg-[#111111] border border-[#222] space-y-3">
              <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                <MessageSquare className="w-5 h-5" />
              </div>
              <h3 className="font-semibold text-white">เวลาทำการ</h3>
              <p className="text-xs text-[#8E8E8E]">จันทร์ - ศุกร์ (09:00 - 18:00 น.)</p>
            </div>
          </div>

          <div className="md:col-span-2 p-6 rounded-2xl bg-[#111111] border border-[#222] space-y-6">
            <h2 className="text-xl font-bold">ส่งข้อความติดต่อทีมงาน</h2>

            {submitted ? (
              <div className="p-6 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 space-y-2 text-center">
                <CheckCircle2 className="w-10 h-10 mx-auto text-emerald-400" />
                <h3 className="font-semibold text-lg">ส่งข้อความสำเร็จแล้ว!</h3>
                <p className="text-xs text-emerald-300/80">ทีมงานได้รับข้อความของท่านเรียบร้อยแล้ว และจะติดต่อกลับทางอีเมลโดยเร็วที่สุด</p>
              </div>
            ) : (
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label className="block text-xs font-medium text-[#A09890] mb-1">ชื่อของคุณ</label>
                  <input
                    type="text"
                    required
                    placeholder="กรอกชื่อ-นามสกุล"
                    className="w-full px-4 py-3 rounded-xl bg-[#1E1B18] border border-[#36322E] text-white focus:outline-none focus:border-[#F97316] text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-[#A09890] mb-1">อีเมลติดต่อ</label>
                  <input
                    type="email"
                    required
                    placeholder="yourname@example.com"
                    className="w-full px-4 py-3 rounded-xl bg-[#1E1B18] border border-[#36322E] text-white focus:outline-none focus:border-[#F97316] text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-[#A09890] mb-1">รายละเอียดข้อความ/ปัญหาที่พบ</label>
                  <textarea
                    rows={4}
                    required
                    placeholder="อธิบายปัญหาที่ต้องการความช่วยเหลือ..."
                    className="w-full px-4 py-3 rounded-xl bg-[#1E1B18] border border-[#36322E] text-white focus:outline-none focus:border-[#F97316] text-sm"
                  ></textarea>
                </div>
                <button
                  type="submit"
                  className="w-full py-3.5 rounded-xl bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-semibold text-sm flex items-center justify-center gap-2 transition-all"
                >
                  <Send className="w-4 h-4" />
                  <span>ส่งข้อความ</span>
                </button>
              </form>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
