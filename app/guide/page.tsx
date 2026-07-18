import type { Metadata } from "next";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export const metadata: Metadata = {
  title: "Audio Guide — HarmoniQ",
  description: "คู่มือสำหรับมือใหม่เรื่องย่านความถี่เสียง EQ และ Compressor parameter เพื่อให้ปรับแต่งเสียงได้อย่างมืออาชีพ",
};

// พารามิเตอร์ Compressor แต่ละตัวพร้อมคำอธิบาย
const PARAMS = [
  { name: "Threshold", unit: "dBFS", desc: "ระดับเสียงที่ Compressor จะเริ่มทำงาน เมื่อเสียงดังเกิน Threshold คอมเพรสเซอร์จะบีบอัดทันที", tip: "ตั้งต่ำ (เช่น -20 dBFS) = กดเสียงหนักขึ้น, ตั้งสูง (เช่น -6 dBFS) = ผ่อนโยนกว่า" },
  { name: "Ratio", unit: "x:1", desc: "อัตราส่วนการบีบอัดเสียง เช่น 4:1 แปลว่าเสียงที่เกิน Threshold 4 dB จะถูกบีบเหลือ 1 dB", tip: "2:1 = gentle, 4:1 = standard, 8:1+ = limiting" },
  { name: "Attack", unit: "ms", desc: "เวลาที่คอมเพรสเซอร์ใช้ในการเริ่มกดเสียงหลังจากที่เสียงดังเกิน Threshold", tip: "Attack เร็ว = จับ transient ได้ดี (ดรัมส์), Attack ช้า = เสียงชัดขึ้นในช่วงต้น" },
  { name: "Release", unit: "ms", desc: "เวลาที่คอมเพรสเซอร์ใช้ในการปล่อยเสียงกลับสู่ปกติหลังจากเสียงเบากว่า Threshold", tip: "Release สั้นเกินไปทำให้เสียง 'หายใจ' (pumping), ยาวเกินไปทำให้เสียงถูกกดค้าง" },
  { name: "Knee", unit: "dB", desc: "ช่วงเปลี่ยนผ่านรอบ Threshold — Soft Knee ทำให้การบีบอัดค่อยๆ เพิ่มขึ้นอย่างราบรื่น", tip: "Soft Knee (6 dB+) = ฟังดูธรรมชาติ, Hard Knee (0 dB) = ตัดชัดเจน" },
  { name: "Makeup Gain", unit: "dB", desc: "การบูสต์ระดับเสียงที่หายไปจากการบีบอัด เพื่อให้เสียง Output ดังเท่ากับ Input", tip: "ใช้หลังจากตั้ง Ratio และ Threshold เสร็จแล้ว" },
];

const EQ_BANDS = [
  { range: "Sub Bass (20–80 Hz)", color: "#7C3AED", desc: "ความรู้สึก 'ฟีล' ของเบส ส่วนใหญ่รู้สึกได้มากกว่าได้ยิน ตัดออกถ้าเสียงโคลงและอ้วน" },
  { range: "Bass (80–250 Hz)", color: "#A78BFA", desc: "น้ำหนักของเบสและกีตาร์เบส บูสต์เพื่อความอบอุ่น คัตถ้า 'โคลน'" },
  { range: "Low-Mid (250–2000 Hz)", color: "#E5A93D", desc: "ย่านที่ทำให้เสียง 'มัว' ถ้าบูสต์มากเกิน คัตเบาๆ เพื่อความชัดเจน" },
  { range: "Upper-Mid (2000–6000 Hz)", color: "#F59E0B", desc: "ความชัดของเสียงร้องและกีตาร์ บูสต์เพื่อ Presence คัตถ้า 'แหลมแทง'" },
  { range: "Presence (6000–12000 Hz)", color: "#22D3EE", desc: "ความสดใสและรายละเอียดเสียง บูสต์เพื่อ 'ชีวิตชีวา' คัตถ้า 'แสบหู'" },
  { range: "Air (12000–20000 Hz)", color: "#34D399", desc: "ความโปร่งเบาสบาย บูสต์เบาๆ ทำให้เสียงร้องดูสด แต่ระวัง Sibilance" },
];

export default function GuidePage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col">
      <Navbar />
      <main className="flex-grow mx-auto w-full max-w-5xl px-4 py-12 space-y-16">
        <header className="space-y-3">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#E5A93D] uppercase">Audio Guide</p>
          <h1 className="text-4xl font-bold tracking-tight">คู่มือการปรับแต่งเสียง</h1>
          <p className="text-[#8E8E8E] max-w-2xl leading-relaxed">
            เข้าใจหลักการ EQ และ Compressor เพื่อนำมาใช้งานใน HarmoniQ ได้อย่างถูกต้องและได้ผลลัพธ์ที่ดีที่สุด
          </p>
        </header>

        {/* ย่านความถี่ EQ */}
        <section className="space-y-5">
          <h2 className="text-2xl font-bold">ย่านความถี่ EQ</h2>
          <div className="space-y-3">
            {EQ_BANDS.map((band) => (
              <div key={band.range} className="flex gap-4 items-start rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] p-4">
                <span
                  className="mt-1 w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ background: band.color, boxShadow: `0 0 8px ${band.color}80` }}
                />
                <div>
                  <p className="text-sm font-semibold text-[#E0E0E0] mb-1">{band.range}</p>
                  <p className="text-sm text-[#888888] leading-relaxed">{band.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* พารามิเตอร์ Compressor */}
        <section className="space-y-5">
          <h2 className="text-2xl font-bold">พารามิเตอร์ Compressor</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            {PARAMS.map((p) => (
              <div key={p.name} className="rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] p-5 space-y-2">
                <div className="flex items-baseline gap-2">
                  <h3 className="text-base font-bold text-[#E5A93D]">{p.name}</h3>
                  <span className="text-xs text-[#555555] font-mono">{p.unit}</span>
                </div>
                <p className="text-sm text-[#AAAAAA] leading-relaxed">{p.desc}</p>
                <p className="text-xs text-[#666666] border-t border-[#1E1E1E] pt-2 leading-relaxed">💡 {p.tip}</p>
              </div>
            ))}
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
