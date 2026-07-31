"use client";
import React from "react";
import Link from "next/link";
import { Navbar } from "./components/Navbar";
import { Footer } from "./components/Footer";
import HowItWorks from "./components/HowItWorks";
import FaqSection from "./components/FaqSection";
import {
  Music, Sliders, Zap, Disc, ArrowRight, CheckCircle2, ShieldCheck, Sparkles,
  Cpu, HardDrive, Lock, BrainCircuit
} from "lucide-react";

const STATS = [
  { number: "99.9%", label: "AI Accuracy Rate", icon: BrainCircuit },
  { number: "24-bit", label: "Lossless Export", icon: HardDrive },
  { number: "< 2 ชม.", label: "Auto File Deletion", icon: Lock },
  { number: "4-Stem", label: "Independent Tracks", icon: Cpu },
];

const FEATURES = [
  {
    icon: Music, title: "4-Stem Separation", desc: "แยกเพลงเป็น 4 แทร็กอิสระ: Vocals, Drums, Bass, และ Other ด้วยความแม่นยำสูง", color: "purple",
  },
  {
    icon: Sliders, title: "Smart AutoEQ", desc: "ปรับแต่งย่านความถี่ด้วย AI โมเดล CNN & LSTM ให้เสียงใสดียิ่งขึ้นโดยอัตโนมัติ", color: "blue",
  },
  {
    icon: Zap, title: "Studio Compressor", desc: "ควบคุมไดนามิกเสียง ปรับ Threshold, Ratio, Attack, Release และ Knee ได้อย่างอิสระ", color: "indigo",
  },
  {
    icon: Disc, title: "Pitch Shift & Key", desc: "ปรับระดับคีย์เพลง ±12 Semitones โดยไม่เสียจังหวะ พร้อมตรวจจับ Tempo & Key อัตโนมัติ", color: "pink",
  },
];

const COLOR_MAP: Record<string, { border: string; bg: string; text: string; hover: string }> = {
  purple: { border: "border-purple-500/20", bg: "bg-purple-500/10", text: "text-purple-400", hover: "hover:border-purple-500/40" },
  blue: { border: "border-blue-500/20", bg: "bg-blue-500/10", text: "text-blue-400", hover: "hover:border-blue-500/40" },
  indigo: { border: "border-indigo-500/20", bg: "bg-indigo-500/10", text: "text-indigo-400", hover: "hover:border-indigo-500/40" },
  pink: { border: "border-pink-500/20", bg: "bg-pink-500/10", text: "text-pink-400", hover: "hover:border-pink-500/40" },
};

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col bg-[#0D0B0A] text-[#F5F0EB]">
      <Navbar />

      <main className="flex-grow">
        {/* ──────── Hero Section ──────── */}
        <section className="relative overflow-hidden pt-16 pb-24 md:pt-24 md:pb-32 border-b border-[#1E1E1E]">
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-purple-600/15 blur-[120px] rounded-full pointer-events-none animate-glow-pulse" />
          <div className="absolute top-1/3 left-1/3 w-[400px] h-[250px] bg-blue-600/10 blur-[100px] rounded-full pointer-events-none animate-glow-pulse" style={{ animationDelay: "1.5s" }} />

          <div className="max-w-6xl mx-auto px-4 text-center relative z-10 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs md:text-sm font-medium">
              <Sparkles className="w-4 h-4" />
              <span>Next-Generation AI Audio Separation & Mastering</span>
            </div>

            <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight leading-tight max-w-4xl mx-auto">
              <span className="inline-block">แยกแทร็กเสียงดนตรี</span>{" "}
              <span className="inline-block">และมาสเตอริ่ง</span>{" "}
              <span className="inline-block bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                ด้วย AI อัจฉริยะ
              </span>
            </h1>

            <p className="text-base md:text-xl text-[#A09890] max-w-2xl mx-auto font-light leading-relaxed">
              เครื่องมือสำหรับ Producer และนักดนตรี แยก เสียงร้อง, กลอง, เบส และเครื่องดนตรีออกจากกันด้วย PyTorch AI พร้อมปรับ EQ, Compressor และ Pitch Shift ในที่เดียว
            </p>

            <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/studio"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-semibold flex items-center justify-center gap-2 shadow-[0_4px_20px_rgba(249,115,22,0.25)] transition-all transform hover:-translate-y-0.5"
              >
                <span>เข้าสู่ Studio Workspace</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/pricing"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-[#161412] border border-[#2C2824] hover:border-[#36322E] text-[#F5F0EB] hover:text-white font-semibold transition-all"
              >
                ดูราคาและแพ็กเกจ
              </Link>
            </div>

            <div className="pt-8 flex flex-wrap items-center justify-center gap-6 text-xs text-[#A09890]">
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

        {/* ──────── Social Proof / Metrics Bar ──────── */}
        <section className="py-14 bg-[#0D0B0A] border-b border-[#1E1E1E]">
          <div className="max-w-6xl mx-auto px-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {STATS.map((stat) => {
                const Icon = stat.icon;
                return (
                  <div key={stat.label} className="text-center space-y-2">
                    <div className="w-10 h-10 rounded-xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mx-auto text-purple-400">
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="text-2xl font-bold text-white">{stat.number}</div>
                    <div className="text-xs text-[#A09890]">{stat.label}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* ──────── Features Grid ──────── */}
        <section className="py-20 bg-[#0D0B0A]">
          <div className="max-w-6xl mx-auto px-4 space-y-12">
            <div className="text-center space-y-3">
              <h2 className="text-3xl md:text-4xl font-bold">ฟีเจอร์ระดับมืออาชีพสำหรับโปรดิวเซอร์</h2>
              <p className="text-[#A09890] max-w-xl mx-auto text-sm md:text-base">
                เครื่องมือครบครันที่ช่วยให้การแยกชิ้นดนตรีและการปรับแต่งเสียงเป็นเรื่องง่ายและรวดเร็ว
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {FEATURES.map((f) => {
                const c = COLOR_MAP[f.color];
                const Icon = f.icon;
                return (
                  <div key={f.title} className={`p-6 rounded-2xl bg-[#161412] border ${c.border} ${c.hover} transition-all space-y-4 group`}>
                    <div className={`w-12 h-12 rounded-xl ${c.bg} border ${c.border} flex items-center justify-center ${c.text} transition-all group-hover:scale-110`}>
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-lg font-semibold">{f.title}</h3>
                    <p className="text-sm text-[#A09890] leading-relaxed">{f.desc}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* ──────── How It Works ──────── */}
        <HowItWorks />

        {/* ──────── AI Model Specs & Security ──────── */}
        <section className="py-20 bg-[#0D0B0A] border-t border-[#1E1E1E]">
          <div className="max-w-6xl mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-10 items-center">
              <div className="space-y-5">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium">
                  <Cpu className="w-4 h-4" />
                  <span>AI Model Specs</span>
                </div>
                <h2 className="text-3xl font-bold tracking-tight">Powered by Open-Unmix PyTorch</h2>
                <p className="text-sm text-[#A09890] leading-relaxed">
                  HarmoniQ ใช้โมเดล Open-Unmix ที่ปรับแต่งเพิ่มเติมบน PyTorch เพื่อแยกแทร็กเสียงด้วยคุณภาพสูงสุด พร้อมระบบ Zero Data Retention — ไฟล์ของคุณจะถูกลบอัตโนมัติภายใน 2 ชั่วโมงหลังประมวลผล
                </p>
                <ul className="space-y-3 text-sm">
                  {[
                    "CNN + LSTM Neural Network สำหรับ AutoEQ",
                    "Real-time Pitch Shift ด้วย PSOLA Algorithm",
                    "24-bit Lossless WAV Export",
                    "TLS Encryption ทุกการเชื่อมต่อ",
                  ].map((item) => (
                    <li key={item} className="flex items-start gap-2 text-[#A09890]">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="hidden md:flex items-center justify-center">
                <div className="w-full max-w-sm aspect-square rounded-3xl bg-gradient-to-br from-purple-600/10 via-indigo-600/5 to-pink-600/10 border border-[#2C2824] flex items-center justify-center p-8">
                  <div className="text-center space-y-4">
                    <BrainCircuit className="w-16 h-16 text-purple-400 mx-auto opacity-60" />
                    <p className="text-xs text-[#5C5854] font-mono">Open-Unmix v2.1 · PyTorch Backend</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ──────── FAQ ──────── */}
        <FaqSection />

        {/* ──────── Final CTA ──────── */}
        <section className="py-20 bg-[#0D0B0A] border-t border-[#1E1E1E]">
          <div className="max-w-4xl mx-auto px-4 text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight">พร้อมสัมผัสประสบการณ์แยกเสียงด้วย AI แล้วหรือยัง?</h2>
            <p className="text-sm md:text-base text-[#A09890] max-w-2xl mx-auto">
              เริ่มต้นใช้งานได้ทันที 3 เพลงต่อเดือนในแพ็กเกจ Free หรืออัปเกรดเพื่อการใช้งานที่ไม่จำกัด
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-2">
              <Link
                href="/studio"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-semibold flex items-center justify-center gap-2 shadow-[0_4px_20px_rgba(249,115,22,0.25)] transition-all transform hover:-translate-y-0.5"
              >
                <span>เริ่มใช้งานฟรี</span>
                <ArrowRight className="w-5 h-5" />
              </Link>
              <Link
                href="/pricing"
                className="w-full sm:w-auto px-8 py-4 rounded-xl bg-[#161412] border border-[#2C2824] hover:border-[#36322E] text-[#F5F0EB] hover:text-white font-semibold transition-all"
              >
                เปรียบเทียบแพ็กเกจ
              </Link>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}