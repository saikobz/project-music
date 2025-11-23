import UploadBox from "./components/UploadBox";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0F0B1D] via-[#1C162C] to-[#7C3AED] text-[#EDE9FE]">
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-10">
        <header className="flex flex-col gap-3">
          <span className="inline-flex items-center gap-2 self-start rounded-full bg-[#7C3AED] px-4 py-1 text-sm font-semibold uppercase tracking-wide text-white">
            HarmoniQ
          </span>
          <h1 className="text-4xl md:text-5xl font-bold leading-tight">
            HarmoniQ – AI Music Stem &amp; Mastering Toolkit
          </h1>
          <p className="text-lg text-[#EDE9FE]/80 max-w-3xl">
            อัปโหลดไฟล์ WAV เพื่อแยกสเต็ม ปรับ EQ/Compressor/ Pitch และวิเคราะห์คีย์กับจังหวะได้ในที่เดียว พร้อมตัวเล่นหลายสเต็มบนหน้าเดียว
          </p>
        </header>

        <section className="bg-[#1C162C] border border-[#7C3AED]/30 rounded-2xl shadow-2xl backdrop-blur-lg">
          <UploadBox />
        </section>
      </div>
    </main>
  );
}
