import type { Metadata } from "next";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export const metadata: Metadata = {
  title: "AI Models — HarmoniQ",
  description: "เรียนรู้เบื้องหลังโมเดล AI ที่ใช้ใน HarmoniQ: Open-Unmix สำหรับแยกสเตมเสียง และ Auto-EQ (CNN/LSTM) สำหรับปรับแต่งคุณภาพเสียงอัตโนมัติ",
};

// ข้อมูลโมเดลแต่ละตัวที่ระบบใช้งาน
const MODELS = [
  {
    name: "Open-Unmix (UMX)",
    tag: "Stem Separation",
    color: "#A78BFA",
    bg: "rgba(167,139,250,0.08)",
    border: "rgba(167,139,250,0.2)",
    desc: "โมเดล deep learning แบบ Recurrent Neural Network (BLSTM) ที่ได้รับการเทรนบนชุดข้อมูล MUSDB18 ทำหน้าที่แยกเสียงดนตรีออกเป็น 4 แทร็กอิสระ ได้แก่ Vocals, Drums, Bass และ Other",
    specs: [
      { label: "Architecture", value: "Bidirectional LSTM" },
      { label: "Dataset", value: "MUSDB18 (150 tracks)" },
      { label: "Output Stems", value: "4 (Vocals, Drums, Bass, Other)" },
      { label: "Framework", value: "PyTorch via open-unmix" },
    ],
  },
  {
    name: "Auto-EQ CNN",
    tag: "Auto EQ",
    color: "#22D3EE",
    bg: "rgba(34,211,238,0.08)",
    border: "rgba(34,211,238,0.2)",
    desc: "โมเดล Convolutional Neural Network (CNN) ที่วิเคราะห์สเปกตรัมเสียงแล้วทำนายค่าการบูสต์/ลดย่านความถี่ที่เหมาะสมตาม Genre Profile ที่เลือก",
    specs: [
      { label: "Architecture", value: "1D CNN (cnn-v1)" },
      { label: "Input", value: "Mel Spectrogram" },
      { label: "Output", value: "EQ gain per band (dB)" },
      { label: "Latency", value: "~0.5s per track" },
    ],
  },
  {
    name: "Auto-EQ LSTM",
    tag: "Auto EQ",
    color: "#22D3EE",
    bg: "rgba(34,211,238,0.06)",
    border: "rgba(34,211,238,0.15)",
    desc: "โมเดล LSTM แบบ sequence-aware ที่มองลำดับเวลาของสเปกตรัมเสียง ทำให้จับ Dynamic ของเสียงได้แม่นยำกว่า CNN ในไฟล์เสียงที่มีการเปลี่ยนแปลงตลอดเวลา",
    specs: [
      { label: "Architecture", value: "LSTM (lstm-last)" },
      { label: "Input", value: "Mel Spectrogram (sequence)" },
      { label: "Output", value: "EQ gain per band (dB)" },
      { label: "Latency", value: "~1.2s per track" },
    ],
  },
];

export default function ModelsPage() {
  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col">
      <Navbar />
      <main className="flex-grow mx-auto w-full max-w-5xl px-4 py-12 space-y-12">
        <header className="space-y-3">
          <p className="text-xs font-semibold tracking-[0.2em] text-[#A78BFA] uppercase">AI Engine</p>
          <h1 className="text-4xl font-bold tracking-tight">โมเดล AI ที่ใช้ใน HarmoniQ</h1>
          <p className="text-[#8E8E8E] max-w-2xl leading-relaxed">
            HarmoniQ ขับเคลื่อนด้วยโมเดล Machine Learning เฉพาะทางด้านเสียงดนตรี ได้รับการเทรนและปรับแต่งอย่างละเอียดสำหรับงาน Source Separation และ Automatic Equalization
          </p>
        </header>

        <div className="space-y-6">
          {MODELS.map((model) => (
            <div
              key={model.name}
              className="rounded-xl border p-6 space-y-4"
              style={{ background: model.bg, borderColor: model.border }}
            >
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <span className="text-xs font-semibold tracking-widest uppercase mb-1 block" style={{ color: model.color }}>
                    {model.tag}
                  </span>
                  <h2 className="text-xl font-bold text-[#F3F3F3]">{model.name}</h2>
                </div>
              </div>
              <p className="text-sm text-[#AAAAAA] leading-relaxed">{model.desc}</p>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {model.specs.map((s) => (
                  <div key={s.label} className="rounded-lg bg-[#111111] border border-[#1E1E1E] p-3">
                    <p className="text-[10px] font-semibold text-[#555555] uppercase tracking-wider mb-1">{s.label}</p>
                    <p className="text-sm font-medium text-[#E0E0E0]">{s.value}</p>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
}
