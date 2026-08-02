"use client";

import React, { useState } from "react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import UploadBox from "../components/UploadBox";

export default function StudioPage() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`flex min-h-screen flex-col bg-[#0D0B0A] text-[#F5F0EB] transition-all duration-300 ${
      isExpanded ? "overflow-y-auto" : "md:h-screen md:overflow-hidden overflow-y-auto"
    }`}>
      <Navbar />

      <main className={`mx-auto flex-grow w-full px-4 py-4 md:py-6 space-y-4 md:space-y-6 flex flex-col justify-center transition-all duration-300 ${
        isExpanded ? "max-w-7xl" : "max-w-5xl"
      }`}>
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-[#F97316] uppercase tracking-widest">
            <span className="w-2 h-2 rounded-full bg-[#F97316] animate-pulse"></span>
            HarmoniQ Studio Workspace
          </div>
          <h1 className="text-3xl md:text-4xl font-bold leading-tight">
            AI Stem Separator &amp; Audio Mastering
          </h1>
          <p className="text-sm md:text-base text-[#8E8E8E] max-w-3xl font-light">
            อัปโหลดไฟล์ WAV เพื่อแยกเสียงดนตรีด้วย Open-Unmix AI พร้อมปรับ EQ, Compressor, Pitch Shift และการวิเคราะห์ Tempo/Key
          </p>
        </header>

        <section className="bg-[#161412] border border-[#2C2824] rounded-2xl shadow-2xl overflow-hidden">
          <UploadBox onHeightChange={setIsExpanded} />
        </section>
      </main>

      <Footer />
    </div>
  );
}
