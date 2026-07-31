"use client";
import React from "react";
import { Upload, Cpu, Download, ArrowRight } from "lucide-react";

const STEPS = [
  {
    icon: Upload,
    title: "Upload Audio",
    description: "ลากวางไฟล์ WAV หรือ MP3 ของคุณขึ้นไปยัง Studio Workspace",
    color: "purple",
  },
  {
    icon: Cpu,
    title: "AI Processing & FX",
    description: "PyTorch AI แยก 4 แทร็กอิสระ พร้อมปรับ EQ, Compressor และ Pitch Shift อัตโนมัติ",
    color: "indigo",
  },
  {
    icon: Download,
    title: "Export Lossless",
    description: "ดาวน์โหลดแทร็กแยกชิ้นคุณภาพสูง 24-bit WAV พร้อมใช้งานในโปรเจกต์ของคุณ",
    color: "pink",
  },
] as const;

const COLOR_CONFIG = {
  purple: { border: "border-purple-500/20", bg: "bg-purple-500/10", text: "text-purple-400", glow: "shadow-purple-500/10" },
  indigo: { border: "border-indigo-500/20", bg: "bg-indigo-500/10", text: "text-indigo-400", glow: "shadow-indigo-500/10" },
  pink: { border: "border-pink-500/20", bg: "bg-pink-500/10", text: "text-pink-400", glow: "shadow-pink-500/10" },
};

export default function HowItWorks() {
  return (
    <section className="py-24 bg-[#0D0B0A] overflow-hidden">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center space-y-3 mb-16">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
            วิธีการทํางาน
          </h2>
          <p className="text-[#A09890] max-w-xl mx-auto text-sm md:text-base">
            เพียง 3 ขั้นตอนง่าย ๆ ก็ได้แทร็กเสียงพร้อมใช้งานในโปรเจกต์ของคุณ
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-4 lg:gap-6">
          {STEPS.map((step, i) => {
            const colors = COLOR_CONFIG[step.color];
            const Icon = step.icon;

            return (
              <div key={step.title} className="relative flex flex-col items-center text-center group">
                {/* Arrow connector between steps on desktop */}
                {i < STEPS.length - 1 && (
                  <div className="hidden md:flex absolute top-6 -right-3 lg:-right-4 z-20 items-center justify-center">
                    <div className="w-7 h-7 rounded-full bg-[#181818] border border-[#2E2E2E] flex items-center justify-center shadow-lg">
                      <ArrowRight className="w-3.5 h-3.5 text-[#666666]" />
                    </div>
                  </div>
                )}

                {/* Step icon with numbered badge */}
                <div className={`relative z-10 w-20 h-20 rounded-2xl ${colors.bg} border ${colors.border} flex items-center justify-center ${colors.text} mb-6 transition-all duration-300 group-hover:scale-110 group-hover:shadow-xl ${colors.glow}`}>
                  <Icon className="w-8 h-8" />
                  <span className={`absolute -top-2.5 -right-2.5 w-7 h-7 rounded-full ${colors.bg} border-2 border-[#0D0B0A] ${colors.border} flex items-center justify-center text-xs font-bold ${colors.text}`}>
                    {i + 1}
                  </span>
                </div>

                {/* Content card */}
                <div className={`w-full p-5 rounded-xl border transition-all duration-300 ${colors.border} ${colors.bg} group-hover:border-opacity-60 group-hover:shadow-lg ${colors.glow}`}>
                  <h3 className="text-base font-semibold text-white mb-2">{step.title}</h3>
                  <p className="text-sm text-[#A09890] leading-relaxed">
                    {step.description}
                  </p>
                </div>

                {/* Mobile downward arrow */}
                {i < STEPS.length - 1 && (
                  <div className="md:hidden flex items-center justify-center py-2">
                    <ArrowRight className="w-5 h-5 text-[#444444] rotate-90" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}