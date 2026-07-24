"use client";
import { useState } from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import CheckoutModal from "../components/CheckoutModal";

export default function PricingPage() {
  const [selectedTier, setSelectedTier] = useState<"BASIC" | "PRO" | null>(null);

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow mx-auto w-full max-w-5xl px-4 py-16">
        <div className="text-center mb-12 space-y-3">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#34D399] uppercase">Subscription Plans</p>
          <h1 className="text-4xl font-extrabold tracking-tight"> HarmoniQ Plans & Pricing</h1>
          <p className="text-[#8E8E8E] max-w-2xl mx-auto text-sm leading-relaxed">
            เลือกแพ็กเกจที่เหมาะสำหรับการแยกแทร็กเสียง และการมิกซ์เสียงด้วย AI AutoEQ (LSTM & CNN)
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Free Plan */}
          <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 flex flex-col justify-between hover:border-[#333333] transition-colors">
            <div>
              <h3 className="text-xl font-bold mb-2 text-[#34D399]">Free</h3>
              <div className="text-3xl font-extrabold mb-4 text-[#F3F3F3]">0 THB</div>
              <ul className="text-[#8E8E8E] text-sm space-y-3 mb-6">
                <li className="flex items-center gap-2"><span className="text-[#34D399]">✓</span> 1 เพลง / เดือน (ความยาวไม่เกิน 3 นาที)</li>
                <li className="flex items-center gap-2"><span className="text-[#34D399]">✓</span> AutoEQ - โมเดล LSTM</li>
                <li className="flex items-center gap-2 text-[#555555] line-through"><span>✗</span> AutoEQ - โมเดล CNN (Locked)</li>
                <li className="flex items-center gap-2 text-[#555555] line-through"><span>✗</span> AI Auto Mastering</li>
              </ul>
            </div>
            <button className="w-full py-2.5 bg-[#1A1A1A] text-[#666666] font-medium rounded-lg cursor-not-allowed">
              แพ็กเกจปัจจุบัน
            </button>
          </div>

          {/* Basic Plan */}
          <div className="bg-[#111111] border border-[#34D399]/40 rounded-2xl p-6 flex flex-col justify-between relative shadow-lg shadow-[#34D399]/5">
            <span className="absolute -top-3 right-6 bg-[#34D399] text-[#0A0A0A] text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full">
              Popular
            </span>
            <div>
              <h3 className="text-xl font-bold mb-2 text-[#F3F3F3]">Basic</h3>
              <div className="text-3xl font-extrabold mb-4 text-[#F3F3F3]">99 THB<span className="text-sm font-normal text-[#8E8E8E]">/เดือน</span></div>
              <ul className="text-[#CCCCCC] text-sm space-y-3 mb-6">
                <li className="flex items-center gap-2"><span className="text-[#34D399]">✓</span> 15 เพลง / เดือน</li>
                <li className="flex items-center gap-2"><span className="text-[#34D399]">✓</span> AutoEQ ได้ทั้งโมเดล **LSTM & CNN**</li>
                <li className="flex items-center gap-2"><span className="text-[#34D399]">✓</span> Export ไฟล์ WAV Lossless</li>
                <li className="flex items-center gap-2 text-[#555555] line-through"><span>✗</span> AI Auto Mastering</li>
              </ul>
            </div>
            <button
              onClick={() => setSelectedTier("BASIC")}
              className="w-full py-2.5 bg-[#34D399] hover:bg-[#2cb984] font-semibold text-[#0A0A0A] rounded-lg transition-colors cursor-pointer"
            >
              สมัครแพ็กเกจ Basic
            </button>
          </div>

          {/* Pro Plan */}
          <div className="bg-gradient-to-b from-[#1A162B] to-[#111111] border border-purple-500/40 rounded-2xl p-6 flex flex-col justify-between">
            <div>
              <h3 className="text-xl font-bold mb-2 text-purple-400">Pro</h3>
              <div className="text-3xl font-extrabold mb-4 text-[#F3F3F3]">299 THB<span className="text-sm font-normal text-[#8E8E8E]">/เดือน</span></div>
              <ul className="text-[#CCCCCC] text-sm space-y-3 mb-6">
                <li className="flex items-center gap-2"><span className="text-purple-400">✓</span> **ไม่จำกัดจำนวนเพลง (Unlimited)**</li>
                <li className="flex items-center gap-2"><span className="text-purple-400">✓</span> ใช้ได้ทุกโมเดล (LSTM & CNN)</li>
                <li className="flex items-center gap-2"><span className="text-purple-400">✓</span> Export ไฟล์ WAV Lossless (High-bitrate)</li>
                <li className="flex items-center gap-2"><span className="text-purple-400">✓</span> **AI Auto Mastering แบบจัดเต็ม**</li>
              </ul>
            </div>
            <button
              onClick={() => setSelectedTier("PRO")}
              className="w-full py-2.5 bg-gradient-to-r from-purple-500 to-indigo-500 hover:opacity-90 font-semibold text-white rounded-lg transition-opacity cursor-pointer"
            >
              สมัครแพ็กเกจ Pro
            </button>
          </div>
        </div>
      </main>

      <Footer />

      {selectedTier && (
        <CheckoutModal
          isOpen={!!selectedTier}
          onClose={() => setSelectedTier(null)}
          tier={selectedTier}
          price={selectedTier === "PRO" ? 299 : 99}
        />
      )}
    </div>
  );
}
