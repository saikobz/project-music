import type { Metadata } from "next";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export const metadata: Metadata = {
  title: "API & Pricing — HarmoniQ",
  description: "เอกสาร API และตารางราคาบริการ HarmoniQ สำหรับนักพัฒนาที่ต้องการเชื่อมต่อโมเดล AI แยกเสียงดนตรีในโปรเจกต์ของตัวเอง",
};

// แผนบริการ Free และ Pro
const PLANS = [
  {
    name: "Free",
    price: "ฟรี",
    color: "#34D399",
    features: [
      "อัปโหลดไฟล์ WAV สูงสุด 100 MB",
      "Stem Separation (4 แทร็ก)",
      "Auto EQ (CNN + LSTM)",
      "Compressor & Pitch Shift",
      "Export WAV / MP3",
      "ไม่มีบัญชี ไม่ต้องล็อกอิน",
    ],
    missing: ["Priority Processing Queue", "Batch Upload", "Commercial License"],
  },
  {
    name: "Pro",
    price: "Coming Soon",
    color: "#E5A93D",
    features: [
      "ทุกอย่างใน Free",
      "Priority Processing Queue",
      "Batch Upload (หลายไฟล์พร้อมกัน)",
      "ขนาดไฟล์สูงสุด 500 MB",
      "Commercial License",
      "API Rate Limit สูงขึ้น",
    ],
    missing: [],
  },
];

// ตัวอย่าง API Code Snippets
const API_EXAMPLES = [
  {
    lang: "cURL",
    code: `curl -X POST http://localhost:8000/separate \\
  -F "file=@/path/to/song.wav" \\
  -F "export_format=wav"`,
  },
  {
    lang: "Python",
    code: `import requests

with open("song.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/separate",
        files={"file": f},
        params={"export_format": "wav"}
    )
data = response.json()
print(data["file_id"], data["zip_url"])`,
  },
];

export default function ApiPricingPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col">
      <Navbar />
      <main className="flex-grow mx-auto w-full max-w-5xl px-4 py-12 space-y-16">
        <header className="space-y-3">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#34D399] uppercase">API & Pricing</p>
          <h1 className="text-4xl font-bold tracking-tight">สำหรับนักพัฒนา</h1>
          <p className="text-[#8E8E8E] max-w-2xl leading-relaxed">
            HarmoniQ ให้บริการ REST API ที่เปิดให้เชื่อมต่อโดยตรง พร้อมเอกสารและตัวอย่างโค้ดสำหรับการ Integrate ในโปรเจกต์ของคุณ
          </p>
        </header>

        {/* แผนบริการ */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold">แผนบริการ</h2>
          <div className="grid sm:grid-cols-2 gap-6">
            {PLANS.map((plan) => (
              <div
                key={plan.name}
                className="rounded-xl border p-6 space-y-4"
                style={{ borderColor: `${plan.color}30`, background: `${plan.color}08` }}
              >
                <div className="flex items-baseline gap-3">
                  <h3 className="text-xl font-bold" style={{ color: plan.color }}>{plan.name}</h3>
                  <span className="text-sm text-[#888888]">{plan.price}</span>
                </div>
                <ul className="space-y-2">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm text-[#CCCCCC]">
                      <span style={{ color: plan.color }}>✓</span> {f}
                    </li>
                  ))}
                  {plan.missing.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm text-[#444444] line-through">
                      <span>✗</span> {f}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* ตัวอย่างการเรียกใช้ API */}
        <section className="space-y-6">
          <h2 className="text-2xl font-bold">ตัวอย่างการเรียกใช้ API</h2>
          <div className="space-y-4">
            {API_EXAMPLES.map((ex) => (
              <div key={ex.lang} className="rounded-xl border border-[#1E1E1E] overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-2 bg-[#111111] border-b border-[#1E1E1E]">
                  <span className="text-xs font-semibold text-[#34D399]">{ex.lang}</span>
                </div>
                <pre className="p-4 text-xs text-[#A0A0A0] font-mono overflow-x-auto leading-relaxed bg-[#080808]">{ex.code}</pre>
              </div>
            ))}
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
