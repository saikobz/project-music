import UploadBox from "./components/UploadBox";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0B1021] via-[#111827] to-[#312E81] text-[#EDE9FE]">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-10">
        <header className="flex flex-col gap-3">
          <span className="inline-flex items-center gap-2 self-start rounded-full bg-[#5B21B6] px-4 py-1 text-sm font-semibold uppercase tracking-wide text-white shadow-lg shadow-purple-900/40">
            HarmoniQ
          </span>
          <h1 className="text-4xl md:text-5xl font-bold leading-tight">
            HarmoniQ — AI Music Stem &amp; Mastering Toolkit
          </h1>
          <p className="text-lg text-[#EDE9FE]/80 max-w-3xl">
            อัปโหลดไฟล์ WAV เพื่อแยกเสียงด้วย AI พร้อม EQ, Compressor, Pitch Shift และการวิเคราะห์
            Tempo/Key/Pitch มีเครื่องเล่นหลายสเตมให้เปิดฟัง ปรับ mute/seek ต่อแทร็ก และดาวน์โหลดผลลัพธ์กลับไปใช้ต่อ
          </p>
        </header>

        <section className="bg-[#111827]/80 border border-[#5B21B6]/30 rounded-2xl shadow-[0_20px_60px_rgba(12,10,26,0.55)] backdrop-blur-lg">
          <UploadBox />
        </section>
      </div>
    </main>
  );
}
