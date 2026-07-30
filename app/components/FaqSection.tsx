"use client";
import React, { useState } from "react";
import { ChevronDown } from "lucide-react";

const FAQS = [
  {
    q: "HarmoniQ รองรับไฟล์เสียงประเภทใดบ้าง?",
    a: "เรารองรับไฟล์ WAV, MP3, FLAC, M4A, OGG, AIFF และอื่น ๆ อีกมากมาย สูงสุดที่ไฟล์ขนาด 100MB สำหรับแพ็กเกจ Free จำกัดที่ 3 เพลงต่อเดือน ส่วน PRO สามารถประมวลผลได้ไม่จำกัด",
  },
  {
    q: "ไฟล์ของฉันจะถูกลบหลังจากประมวลผลหรือไม่?",
    a: "ใช่ ไฟล์ต้นฉบับและผลลัพธ์ทั้งหมดจะถูกลบออกจากเซิร์ฟเวอร์โดยอัตโนมัติภายใน 2 ชั่วโมงหลังจากประมวลผลเสร็จสิ้น เรายังใช้การเข้ารหัส TLS สำหรับการอัปโหลดทุกครั้ง เพื่อให้มั่นใจว่าข้อมูลของคุณปลอดภัย 100%",
  },
  {
    q: "ความแม่นยำของ AI คุ้มค่ากับการใช้งานหรือไม่",
    a: "โมเดล Open-Unmix ที่เราปรับแต่งเพิ่มเติมให้ความแม่นยำสูง โดยเฉพาะการแยก Vocal และ Drums ในเพลงที่มีมิกซ์คุณภาพดี อย่างไรก็ตาม คุณภาพของผลลัพธ์ขึ้นอยู่กับต้นฉบับ — เพลงที่ถูก Master มาอย่างดีจะให้ผลลัพธ์ที่ดีที่สุด",
  },
  {
    q: "ระหว่างแพ็กเกจ Free กับ PRO แตกต่างกันอย่างไร?",
    a: "แพ็กเกจ Free มาพร้อม 3 เพลงต่อเดือน รองรับไฟล์สูงสุด 10MB และความยาวสูงสุด 10 นาที ในขณะที่ PRO ไม่จำกัดจำนวนเพลง รองรับไฟล์สูงสุด 200MB, ความยาวสูงสุด 30 นาที, Export 24-bit WAV, ลำดับคิว優先, และฟีเจอร์ครบถ้วนทั้งหมด",
  },
  {
    q: "สามารถใช้ HarmoniQ เชิงพาณิชย์ได้หรือไม่?",
    a: "ได้ ผลลัพธ์จาก HarmoniQ เป็นของคุณ 100% ไม่มีลิขสิทธิ์ทับซ้อน คุณสามารถนำไปใช้ในโปรเจกต์ ส่วนตัวหรือเชิงพาณิชย์ได้ทันที โดยไม่ต้องระบุแหล่งที่มา",
  },
  {
    q: "มีแผนจะเพิ่มฟีเจอร์อะไรอีกในอนาคต?",
    a: "เรากำลังพัฒนา Mastering Chain อัตโนมัติเต็มรูปแบบ, Reverb/Delay Separation, Real-time Cloud Collaboration, และ API สำหรับนักพัฒนา รวมถึงโมเดล AI ที่รองรับการแยก 8-Stem ในรุ่นถัดไป",
  },
];

export default function FaqSection() {
  const [openIdx, setOpenIdx] = useState<number | null>(0);

  const toggle = (i: number) => setOpenIdx((prev) => (prev === i ? null : i));

  return (
    <section className="py-24 bg-[#0E0E0E] border-t border-[#1F1F1F]">
      <div className="max-w-3xl mx-auto px-4">
        <div className="text-center space-y-3 mb-14">
          <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
            คำถามที่พบบ่อย
          </h2>
          <p className="text-[#A0A0A0] text-sm md:text-base">
            ทุกสิ่งที่คุณต้องการรู้เกี่ยวกับ HarmoniQ
          </p>
        </div>

        <div className="space-y-3">
          {FAQS.map((faq, i) => {
            const isOpen = openIdx === i;
            return (
              <div
                key={i}
                className={`rounded-xl border transition-all ${
                  isOpen
                    ? "bg-[#141414] border-purple-500/20"
                    : "bg-[#0E0E0E] border-[#1F1F1F] hover:border-[#2E2E2E]"
                }`}
              >
                <button
                  onClick={() => toggle(i)}
                  className="w-full flex items-center justify-between gap-3 px-5 py-4 text-left cursor-pointer"
                >
                  <span className="text-sm md:text-base font-medium text-white">
                    {faq.q}
                  </span>
                  <ChevronDown
                    className={`w-4 h-4 text-[#666666] shrink-0 transition-transform duration-200 ${
                      isOpen ? "rotate-180" : ""
                    }`}
                  />
                </button>
                <div
                  className={`overflow-hidden transition-all duration-200 ${
                    isOpen ? "max-h-96" : "max-h-0"
                  }`}
                >
                  <p className="px-5 pb-4 text-sm text-[#A0A0A0] leading-relaxed">
                    {faq.a}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}