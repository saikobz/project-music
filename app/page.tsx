import UploadBox from "./components/UploadBox";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-cyan-900 text-white">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-10">
        <header className="flex flex-col gap-3">
          <span className="inline-flex items-center gap-2 self-start rounded-full bg-white/10 px-4 py-1 text-sm font-semibold uppercase tracking-wide text-cyan-200">
            HarmoniQ
          </span>
          <h1 className="text-4xl md:text-5xl font-bold leading-tight">
            HarmoniQ – AI Music Stem &amp; Mastering Toolkit
          </h1>
          <p className="text-lg text-slate-200 max-w-3xl">
            อัปโหลดไฟล์ WAV เพื่อแยกสเต็ม ปรับ EQ/Compressor/ Pitch และวิเคราะห์คีย์กับจังหวะได้ในที่เดียว พร้อมตัวเล่นหลายสเต็มบนหน้าเดียว
          </p>
        </header>

        <section className="bg-white/5 border border-white/10 rounded-2xl shadow-2xl backdrop-blur-lg">
          <UploadBox />
        </section>
      </div>
    </main>
  );
}
