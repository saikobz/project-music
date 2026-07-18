import type { Metadata } from "next";
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

        {/* Contact Form */}
        <section className="space-y-5">
          <h2 className="text-2xl font-bold">ติดต่อเรา</h2>
          <form className="space-y-4 rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] p-6">
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-[#555555] uppercase tracking-wider mb-1.5">
                  ชื่อ
                </label>
                <input
                  type="text"
                  placeholder="ชื่อของคุณ"
                  className="w-full rounded-lg bg-[#080808] border border-[#1E1E1E] px-3 py-2.5 text-sm text-[#E0E0E0] placeholder-[#333333] focus:border-[#64748B]/60 focus:outline-none transition"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-[#555555] uppercase tracking-wider mb-1.5">
                  อีเมล
                </label>
                <input
                  type="email"
                  placeholder="email@example.com"
                  className="w-full rounded-lg bg-[#080808] border border-[#1E1E1E] px-3 py-2.5 text-sm text-[#E0E0E0] placeholder-[#333333] focus:border-[#64748B]/60 focus:outline-none transition"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs font-semibold text-[#555555] uppercase tracking-wider mb-1.5">
                ข้อความ
              </label>
              <textarea
                rows={4}
                placeholder="พิมพ์ข้อความหรือรายงานปัญหาที่พบ..."
                className="w-full rounded-lg bg-[#080808] border border-[#1E1E1E] px-3 py-2.5 text-sm text-[#E0E0E0] placeholder-[#333333] focus:border-[#64748B]/60 focus:outline-none transition resize-none"
              />
            </div>
            <button
              type="submit"
              className="w-full rounded-lg py-3 text-sm font-bold tracking-widest uppercase bg-gradient-to-r from-[#334155] to-[#64748B] text-white hover:from-[#475569] hover:to-[#94A3B8] transition-all duration-200 cursor-pointer"
            >
              ส่งข้อความ
            </button>
          </form>
        </section>
      </main>
      <Footer />
    </div>
  );
}
