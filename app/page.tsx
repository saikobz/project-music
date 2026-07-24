"use client";

import React from "react";
import Link from "next/link";
import { Navbar } from "./components/Navbar";
import { Footer } from "./components/Footer";
import { Music, Sliders, Zap, Disc, ArrowRight, CheckCircle2, ShieldCheck, Sparkles } from "lucide-react";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[#0A0A0A] text-[#F3F3F3]">
      <Navbar />

      <main className="flex-grow">
        {/* Hero Section */}
        <section className="relative overflow-hidden pt-16 pb-24 md:pt-24 md:pb-32 border-b border-[#1F1F1F]">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-purple-600/15 blur-[120px] rounded-full pointer-events-none" />
          <div className="absolute top-1/3 left-1/3 w-[400px] h-[250px] bg-blue-600/10 blur-[100px] rounded-full pointer-events-none" />

          <div className="max-w-6xl mx-auto px-4 text-center relative z-10 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs md:text-sm font-medium">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span>Next-Generation AI Audio Separation & Mastering</span>
            </div>

            <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight leading-tight max-w-4xl mx-auto">
              <span className="inline-block">แยกแทร็กเสียงดนตรี</span>{" "}
              <span className="inline-block">และมาสเตอริ่ง</span>{" "}
              <span className="inline-block bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                ด้วย AI อัจฉริยะ
              </span>
            </h1>

            <p className="text-base md:text-xl text-[#999999] max-w-2xl mx-auto font-light leading-relaxed">
              เครื่องมือสำหรับ Producer และนักดนตรี แยก เสียงร้อง, กลอง, เบส และเครื่องดนตรีออกจากกันด้วย PyTorch AI พร้อมปรับ EQ, Compressor และ Pitch Shift ในที่เดียว
            </p>

            <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/studio"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-semibold flex items-center justify-center gap-2 shadow-lg shadow-purple-600/25 transition-all transform hover:-translate-y-0.5"
              >
                <span>เข้าสู่ Studio Workspace</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/pricing"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-[#181818] border border-[#2E2E2E] hover:border-[#444444] text-[#E0E0E0] hover:text-white font-semibold transition-all"
              >
                ดูราคาและแพ็กเกจ
              </Link>
            </div>

            <div className="pt-8 flex flex-wrap items-center justify-center gap-6 text-xs text-[#808080]">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>รองรับไฟล์ WAV Lossless</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>ประมวลผลผ่าน Open-Unmix AI</span>
              </div>
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                <span>ความเป็นส่วนตัวสูง ไฟล์ลบอัตโนมัติ</span>
              </div>
            </div>
          </div>
        </section>

        {/* Feature Cards Grid */}
        <section className="py-20 bg-[#0E0E0E]">
          <div className="max-w-6xl mx-auto px-4 space-y-12">
            <div className="text-center space-y-3">
              <h2 className="text-3xl font-bold">ฟีเจอร์ระดับมืออาชีพสำหรับโปรดิวเซอร์</h2>
              <p className="text-[#888888] max-w-xl mx-auto text-sm md:text-base">
                เครื่องมือครบครันที่ช่วยให้การแยกชิ้นดนตรีและการปรับแต่งเสียงเป็นเรื่องง่ายและรวดเร็ว
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="p-6 rounded-2xl bg-[#141414] border border-[#242424] hover:border-purple-500/40 transition-all space-y-4">
                <div className="w-12 h-12 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400">
                  <Music className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-semibold">4-Stem Separation</h3>
                <p className="text-xs text-[#8E8E8E] leading-relaxed">
                  แยกเพลงเป็น 4 แทร็กอิสระ: Vocals, Drums, Bass, และ Other ด้วยความแม่นยำสูง
                </p>
              </div>

              <div className="p-6 rounded-2xl bg-[#141414] border border-[#242424] hover:border-blue-500/40 transition-all space-y-4">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400">
                  <Sliders className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-semibold">Smart AutoEQ</h3>
                <p className="text-xs text-[#8E8E8E] leading-relaxed">
                  ปรับแต่งย่านความถี่ด้วย AI โมเดล CNN &amp; LSTM ให้เสียงใสดียิ่งขึ้นโดยอัตโนมัติ
                </p>
              </div>

              <div className="p-6 rounded-2xl bg-[#141414] border border-[#242424] hover:border-indigo-500/40 transition-all space-y-4">
                <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                  <Zap className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-semibold">Studio Compressor</h3>
                <p className="text-xs text-[#8E8E8E] leading-relaxed">
                  ควบคุมไดนามิกเสียง ปรับ Threshold, Ratio, Attack, Release และ Knee ได้อย่างอิสระ
                </p>
              </div>

              <div className="p-6 rounded-2xl bg-[#141414] border border-[#242424] hover:border-pink-500/40 transition-all space-y-4">
                <div className="w-12 h-12 rounded-xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center text-pink-400">
                  <Disc className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-semibold">Pitch Shift &amp; Key</h3>
                <p className="text-xs text-[#8E8E8E] leading-relaxed">
                  ปรับระดับคีย์เพลง ±12 Semitones โดยไม่เสียจังหวะ พร้อมตรวจจับ Tempo &amp; Key อัตโนมัติ
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-16 border-t border-[#1F1F1F]">
          <div className="max-w-4xl mx-auto px-4 text-center space-y-6">
            <h2 className="text-2xl md:text-4xl font-bold">พร้อมสัมผัสประสบการณ์แยกเสียงด้วย AI แล้วหรือยัง?</h2>
            <p className="text-sm md:text-base text-[#8E8E8E]">
              เริ่มต้นใช้งานได้ทันที 3 เพลงต่อเดือนในแพ็กเกจ Free หรืออัปเกรดเพื่อการใช้งานที่ไม่จำกัด
            </p>
            <div>
              <Link
                href="/studio"
                className="inline-flex items-center gap-2 px-8 py-4 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-semibold transition-all"
              >
                <span>เริ่มใช้งานใน Studio</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
