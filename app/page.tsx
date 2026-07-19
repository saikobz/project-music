"use client";

import React, { useState } from "react";
import { Navbar } from "./components/Navbar";
import { Footer } from "./components/Footer";
import UploadBox from "./components/UploadBox";

// หน้าแรกของระบบ ประกอบจากกล่องอัปโหลดและการแสดงผลเป็นหลัก
export default function Home() {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className={`flex min-h-screen flex-col bg-[#0A0A0A] text-[#F3F3F3] transition-all duration-300 ${
      isExpanded ? "overflow-y-auto" : "md:h-screen md:overflow-hidden overflow-y-auto"
    }`}>
      <Navbar />

      <main className={`mx-auto flex-grow w-full px-4 py-4 md:py-6 space-y-4 md:space-y-6 flex flex-col justify-center transition-all duration-300 ${
        isExpanded ? "max-w-7xl" : "max-w-5xl"
      }`}>
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl md:text-4xl font-bold leading-tight">
            AI Music Stem &amp; Mastering Toolkit
          </h1>
          <p className="text-sm md:text-base text-[#8E8E8E] max-w-3xl font-light">
            อัปโหลดไฟล์ WAV เพื่อแยกเสียงดนตรีด้วย AI พร้อม EQ, Compressor, Pitch Shift และการวิเคราะห์ 
            Tempo/Key/Pitch ปรับแต่แต่งระดับเสียง Mute/Seek/Solo และดาวน์โหลดผลลัพธ์ผ่านเครื่องเล่นสเตมแบบสด
          </p>
        </header>

        <section className="bg-[#121212] border border-[#2A2A2A] rounded-2xl shadow-2xl overflow-hidden">
          <UploadBox onHeightChange={setIsExpanded} />
        </section>
      </main>

      <Footer />
    </div>
  );
}
