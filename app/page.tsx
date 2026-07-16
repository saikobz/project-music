"use client";

import React, { useState } from "react";
import { Navbar } from "./components/Navbar";
import { Footer } from "./components/Footer";
import UploadBox from "./components/UploadBox";
import EqKnowledgePanel from "./components/EqKnowledgePanel";

// หน้าแรกของระบบ ประกอบจากคู่มือ EQ แบบลอย, กล่องอัปโหลด, และการ์ด preset อ้างอิง
export default function Home() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`flex min-h-screen flex-col bg-gradient-to-br from-[#0B1021] via-[#111827] to-[#312E81] text-[#EDE9FE] transition-all duration-300 ${
      isExpanded ? "overflow-y-auto" : "md:h-screen md:overflow-hidden overflow-y-auto"
    }`}>
      <Navbar />
      <EqKnowledgePanel />

      <main className="mx-auto flex-grow w-full max-w-6xl px-4 py-4 md:py-6 space-y-4 md:space-y-6 flex flex-col justify-center">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl md:text-4xl font-bold leading-tight">
            AI Music Stem &amp; Mastering Toolkit
          </h1>
          <p className="text-sm md:text-base text-[#EDE9FE]/80 max-w-3xl">
            อัปโหลดไฟล์ WAV เพื่อแยกเสียงดนตรีด้วย AI พร้อม EQ, Compressor, Pitch Shift และการวิเคราะห์ 
            Tempo/Key/Pitch ปรับแต่แต่งระดับเสียง Mute/Seek/Solo และดาวน์โหลดผลลัพธ์ผ่านเครื่องเล่นสเตมแบบสด
          </p>
        </header>

        <section className="bg-[#111827]/85 border border-[#5B21B6]/30 rounded-2xl shadow-[0_20px_60px_rgba(12,10,26,0.55)] backdrop-blur-lg">
          <UploadBox onHeightChange={setIsExpanded} />
        </section>
      </main>

      <Footer />
    </div>
  );
}
