import type { Metadata } from "next";
import Link from "next/link";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export const metadata: Metadata = {
  title: "About — HarmoniQ",
  description: "เรียนรู้เพิ่มเติมเกี่ยวกับโปรเจกต์ HarmoniQ AI Audio Toolkit และติดต่อทีมผู้พัฒนา",
};

// Tech Stack ที่ระบบใช้งาน
const TECH_STACK = [
  { name: "Next.js 15", role: "Frontend Framework" },
  { name: "FastAPI", role: "Backend API" },
  { name: "PyTorch", role: "AI Inference" },
  { name: "Open-Unmix", role: "Source Separation" },
  { name: "WaveSurfer.js", role: "Waveform Player" },
  { name: "Tailwind CSS 4", role: "Styling" },
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col">
      <Navbar />
      <main className="flex-grow mx-auto w-full max-w-4xl px-4 py-12 space-y-16">

        {/* About Section */}
        <header className="space-y-4">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#64748B] uppercase">About</p>
          <h1 className="text-4xl font-bold tracking-tight">เกี่ยวกับ HarmoniQ</h1>
          <p className="text-[#8E8E8E] max-w-2xl leading-relaxed text-base">
            HarmoniQ เป็นโปรเจกต์วิชาการที่ผสมผสานความสามารถของ Machine Learning
            เข้ากับการออกแบบ Web Application สมัยใหม่ เพื่อให้นักดนตรีและวิศวกรเสียง
            สามารถเข้าถึงเครื่องมือระดับสตูดิโอได้ผ่านเว็บเบราว์เซอร์โดยตรง
          </p>
          <p className="text-[#666666] leading-relaxed text-sm max-w-2xl">
            ระบบ Backend ขับเคลื่อนด้วย <span className="text-[#94A3B8]">FastAPI (Python 3.10)</span> และโมเดล
            PyTorch ฝั่ง Frontend สร้างด้วย <span className="text-[#94A3B8]">Next.js 15 + React 19 + Tailwind CSS 4</span>
          </p>
        </header>

        {/* Tech Stack Grid */}
        <section className="space-y-5">
          <h2 className="text-2xl font-bold">Tech Stack</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {TECH_STACK.map((t) => (
              <div key={t.name} className="rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] p-4">
                <p className="text-sm font-bold text-[#E0E0E0]">{t.name}</p>
                <p className="text-xs text-[#555555] mt-1">{t.role}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Support Link CTA */}
        <section className="rounded-2xl border border-[#1E1E1E] bg-[#0E0E0E] p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-bold">มีข้อสงสัยหรือต้องการติดต่อทีมงาน?</h3>
            <p className="text-xs text-[#8E8E8E] mt-1">สามารถส่งข้อความหรือรายงานปัญหาการใช้งานได้ที่ศูนย์ช่วยเหลือ</p>
          </div>
          <Link
            href="/support"
            className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs rounded-xl transition shrink-0"
          >
            ไปยังศูนย์ช่วยเหลือ &amp; ติดต่อ
          </Link>
        </section>
      </main>
      <Footer />
    </div>
  );
}
