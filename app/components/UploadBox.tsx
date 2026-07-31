// หน้าที่ของไฟล์นี้:
// - เป็นฟอร์มหลักของหน้าเว็บสำหรับรับไฟล์ WAV และเลือกประเภทการประมวลผล
// - เรียก backend ตาม action ที่ผู้ใช้เลือก เช่น แยก stem, Auto-EQ, Compressor และ Pitch Shift
// - จัดการสถานะของ UI เช่น error, ผลวิเคราะห์, ลิงก์ดาวน์โหลด และตัวเล่นผลลัพธ์
"use client";
import React, { useState } from "react";
import { toast } from "sonner";
import WaveformPlayer from "./WaveformPlayer";
import MultiStemLivePlayer from "./MultiStemLivePlayer";
import AudioAnalysis from "./AudioAnalysis";
import { AutoEqSettings } from "./settings/AutoEqSettings";
import { CompressorSettings, DEFAULT_COMPRESSOR_PARAMS, type CompressorParams } from "./settings/CompressorSettings";
import { PitchShiftSettings } from "./settings/PitchShiftSettings";
import ExportMasterModal from "./ExportMasterModal";
import SingleExportModal from "./SingleExportModal";
import { MAX_UPLOAD_BYTES, type AudioAction } from "@/lib/config";
import { useAudioProcessor } from "@/lib/hooks/useAudioProcessor";

// ค่าตั้งต้นของ API และข้อจำกัดการอัปโหลด
const MAX_SIZE_BYTES = MAX_UPLOAD_BYTES;
const AUTO_EQ_DELTA_CLAMP_MIN = 0;
const AUTO_EQ_DELTA_CLAMP_MAX = 6;
const AUTO_EQ_DELTA_CLAMP_DEFAULT = 2;
const AUTO_EQ_MODEL_DEFAULT = "lstm-last";
const AUTO_EQ_MODEL_OPTIONS = [
  { value: "cnn-v1", label: "CNN", hint: "โหมดเดิมของโปรเจกต์" },
  { value: "lstm-last", label: "LSTM", hint: "โมเดลใหม่แบบ sequence-aware" },
];

// ===== Skeleton Components สำหรับแสดงสถานะกำลังประมวลผล (Processing State) =====
function AudioAnalysisSkeleton() {
  return (
    <div className="rounded-xl border border-[#2C2824] bg-[#161412] p-5 shadow-lg animate-pulse">
      <div className="flex items-center gap-2 mb-4">
        <div className="h-5 w-28 bg-[#2C2824] rounded"></div>
        <div className="h-5 w-24 bg-[#2C2824] rounded opacity-50"></div>
      </div>
      <div className="grid grid-cols-3 gap-4 text-center">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-lg bg-[#1E1B18] p-3 border border-[#2C2824] space-y-2 flex flex-col items-center">
            <div className="h-3 w-10 bg-[#2C2824] rounded"></div>
            <div className="h-6 w-16 bg-[#2C2824] rounded"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StemMixerSkeleton() {
  return (
    <div className="rounded-xl border border-[#2C2824] bg-[#161412] p-5 shadow-lg animate-pulse space-y-4">
      <div className="flex items-center justify-between">
        <div className="h-5 w-32 bg-[#2C2824] rounded"></div>
        <div className="flex gap-2">
          <div className="h-8 w-24 bg-[#2C2824] rounded-lg"></div>
          <div className="h-8 w-20 bg-[#2C2824] rounded-lg"></div>
        </div>
      </div>
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="rounded-xl border border-[#2C2824] bg-[#1E1B18] p-3 space-y-3">
            <div className="flex justify-between items-center">
              <div className="space-y-1.5">
                <div className="h-4 w-16 bg-[#2C2824] rounded"></div>
                <div className="h-3 w-28 bg-[#2C2824] rounded opacity-60"></div>
              </div>
              <div className="flex gap-2">
                <div className="h-6 w-16 bg-[#2C2824] rounded-lg"></div>
                <div className="h-6 w-12 bg-[#2C2824] rounded-lg"></div>
              </div>
            </div>
            <div className="h-14 w-full bg-[#161412] rounded border border-[#2C2824] opacity-80 flex items-center px-3">
              <div className="h-2 w-12 bg-[#2C2824] rounded mr-3"></div>
              <div className="h-1.5 flex-grow bg-[#2C2824] rounded"></div>
              <div className="h-4 w-10 bg-[#2C2824] rounded ml-3"></div>
            </div>
            <div className="h-10 w-full bg-[#161412] rounded border border-[#2C2824] opacity-40"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SinglePlayerSkeleton() {
  return (
    <div className="rounded-xl border border-[#2C2824] bg-[#161412] p-5 shadow-lg animate-pulse space-y-4">
      <div className="h-12 w-full bg-[#E5A93D]/10 rounded-lg border border-[#E5A93D]/20 flex items-center justify-center">
        <div className="h-4 w-40 bg-[#E5A93D]/30 rounded"></div>
      </div>
      <div className="h-24 w-full bg-[#1E1B18] rounded-lg border border-[#2C2824] opacity-60"></div>
    </div>
  );
}

interface UploadBoxProps {
  onHeightChange?: (expanded: boolean) => void;
}

function UploadBox({ onHeightChange }: UploadBoxProps) {
  // ===== กลุ่ม state สำหรับค่าที่ผู้ใช้เลือกในฟอร์ม =====
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [action, setAction] = useState<AudioAction>("separate");
  const [compParams, setCompParams] = useState<CompressorParams>(DEFAULT_COMPRESSOR_PARAMS);
  const patchCompParams = (patch: Partial<CompressorParams>) =>
    setCompParams((prev) => ({ ...prev, ...patch }));
  const [genre, setGenre] = useState("pop");
  const [autoEqModel, setAutoEqModel] = useState(AUTO_EQ_MODEL_DEFAULT);
  const [deltaClampDb, setDeltaClampDb] = useState(String(AUTO_EQ_DELTA_CLAMP_DEFAULT));
  const [pitchSteps, setPitchSteps] = useState(0);

  const [isTrimming, setIsTrimming] = useState(false);
  const [trimStart, setTrimStart] = useState("0");
  const [trimEnd, setTrimEnd] = useState("30");
  const [exportFormat, setExportFormat] = useState("wav");

  // C7: business logic ทั้งหมด (upload/process/export/quota/history) อยู่ใน hook นี้
  const {
    loading,
    errorMessage,
    successMessage,
    statusText,
    analysis,
    downloadUrl,
    downloadFileName,
    processingTime,
    fileId,
    zipUrl,
    isExportModalOpen,
    setIsExportModalOpen,
    isExporting,
    isSingleExportModalOpen,
    setIsSingleExportModalOpen,
    setErrorMessage,
    setSuccessMessage,
    setAnalysis,
    setDownloadFileName,
    setFileId,
    setZipUrl,
    setProcessingTime,
    clearDownloadUrl,
    handleUpload,
    handleExport,
    handleSingleExport,
    handleKaraokeDownload,
  } = useAudioProcessor({
    file,
    action,
    exportFormat,
    setExportFormat,
    isTrimming,
    trimStart,
    trimEnd,
    genre,
    autoEqModel,
    deltaClampDb,
    compParams,
    pitchSteps,
  });

  React.useEffect(() => {
    if (onHeightChange) {
      const expanded = action !== "separate" || !!file || loading || !!zipUrl || !!downloadUrl || !!analysis;
      onHeightChange(expanded);
    }
  }, [action, file, loading, zipUrl, downloadUrl, analysis, onHeightChange]);

  React.useEffect(() => {
    // เตือนก่อนปิดหน้าเมื่อกำลังประมวลผล (abort จัดการโดย hook ตอน unmount)
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (loading) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [loading]);

  // ตัวตรวจสอบกลาง ใช้ร่วมกันทั้ง input file และ drag-and-drop
  const handleFileSelect = (selected: File | null) => {
    // ล้างผลลัพธ์ของไฟล์เก่าทั้งหมด เพื่อไม่ให้ค้างอยู่หน้าจอเมื่อเลือกไฟล์ใหม่ (M8)
    setErrorMessage(null);
    setSuccessMessage(null);
    clearDownloadUrl();
    setDownloadFileName(null);
    setFileId(null);
    setZipUrl(null);
    setAnalysis(null);
    setProcessingTime(null);
    setFile(null);
    if (!selected) return;
    const ext = selected.name.toLowerCase().split(".").pop();
    if (ext !== "wav") {
      const msg = "รองรับเฉพาะไฟล์ WAV (.wav) เท่านั้น";
      setErrorMessage(msg);
      toast.error(msg);
      return;
    }
    if (selected.size > MAX_SIZE_BYTES) {
      const msg = "ไฟล์ต้องมีขนาดไม่เกิน 100MB";
      setErrorMessage(msg);
      toast.error(msg);
      return;
    }
    setFile(selected);
    toast.success(`โหลดไฟล์ ${selected.name} สำเร็จ`);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    handleFileSelect(droppedFile || null);
  };

  const eqDeltaClampValue = Number.parseFloat(deltaClampDb);
  const isEqDeltaClampValid =
    Number.isFinite(eqDeltaClampValue) &&
    eqDeltaClampValue >= AUTO_EQ_DELTA_CLAMP_MIN &&
    eqDeltaClampValue <= AUTO_EQ_DELTA_CLAMP_MAX;
  const eqDeltaClampWarning = !isEqDeltaClampValid
    ? `กรอกค่า Delta Clamp ระหว่าง ${AUTO_EQ_DELTA_CLAMP_MIN.toFixed(1)} ถึง ${AUTO_EQ_DELTA_CLAMP_MAX.toFixed(1)} dB`
    : eqDeltaClampValue > 3.5
      ? "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ยิ่งปรับสูง Auto-EQ จะยิ่งแก้หนักขึ้น เสี่ยงโทนแข็ง บวม หรือเสียงแตกได้ง่ายขึ้น"
      : eqDeltaClampValue < 1.0
        ? "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ค่าต่ำจะคุมการปรับไว้เบาและปลอดภัยกว่า แต่ความเปลี่ยนแปลงของเสียงอาจไม่ชัดมาก"
        : "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ค่าช่วงกลางมักบาลานซ์ระหว่างความชัดของผลลัพธ์กับความเป็นธรรมชาติของเสียง";

  const selectedAutoEqModel =
    AUTO_EQ_MODEL_OPTIONS.find((option) => option.value === autoEqModel) ?? AUTO_EQ_MODEL_OPTIONS[0];

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  return (
    <div className={`p-6 text-[#F5F0EB] transition-all duration-500 ${!file ? "flex flex-col items-center justify-center min-h-[50vh]" : "grid gap-6 lg:grid-cols-12"}`}>
      {!file ? (
        <div className="w-full max-w-3xl text-center space-y-6">
          <h2 className="text-3xl font-bold tracking-tight text-[#F5F0EB]">Upload Audio</h2>
          <p className="text-[#8E8E8E] font-medium">Select or drag & drop a WAV file (up to 100MB)</p>
          <label
            className={`mt-6 mx-auto flex h-64 max-w-2xl cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed transition-all
            ${isDragging ? "border-[#E5A93D] bg-[#E5A93D]/10" : "border-[#2C2824] bg-[#161412] hover:border-[#E5A93D]/50 hover:bg-[#1E1B18]"}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              type="file"
              accept="audio/wav,audio/x-wav,.wav"
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
            />
            <div className="text-lg font-medium text-[#F5F0EB]">Click to browse or drag file here</div>
            <div className="mt-2 text-sm text-[#5C5854]">.wav only</div>
          </label>
          {errorMessage && (
            <div className="rounded-lg bg-red-900/30 border border-red-900/50 p-3 text-sm text-red-400 max-w-xl mx-auto">
              {errorMessage}
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="space-y-4 lg:col-span-4">
            <div className="rounded-xl border border-[#2C2824] bg-[#161412] p-4 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-[#8E8E8E] uppercase tracking-wider mb-1">Source File</p>
                  <h2 className="text-lg font-semibold text-[#F5F0EB] truncate max-w-[200px] sm:max-w-[300px]">{file.name}</h2>
                </div>
                <button 
                  onClick={() => handleFileSelect(null)}
                  disabled={loading}
                  className={`text-xs px-3 py-1.5 rounded bg-[#1E1B18] border border-[#2C2824] text-[#8E8E8E] transition ${
                    loading 
                      ? "opacity-50 cursor-not-allowed" 
                      : "hover:text-[#F5F0EB] hover:border-[#8E8E8E] cursor-pointer"
                  }`}
                >
                  Change File
                </button>
              </div>
            </div>

            {/* ===== PROCESSING MODULE – redesigned studio console style ===== */}
            <div className="rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] shadow-xl overflow-hidden">

              {/* ── หัว Module พร้อม indicator LED ── */}
              <div className="flex items-center gap-2.5 px-4 pt-4 pb-3 border-b border-[#1E1E1E]">
                {/* LED indicator กะพริบเมื่อกำลังประมวลผล */}
                <span className={`w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300 ${
                  loading
                    ? "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.6)] animate-pulse"
                    : action === "separate" ? "bg-[#A78BFA] shadow-[0_0_5px_rgba(167,139,250,0.5)]"
                    : action === "eq-ai" ? "bg-[#22D3EE] shadow-[0_0_5px_rgba(34,211,238,0.5)]"
                    : action === "compressor" ? "bg-[#E5A93D] shadow-[0_0_5px_rgba(229,169,61,0.5)]"
                    : "bg-[#34D399] shadow-[0_0_5px_rgba(52,211,153,0.5)]"
                }`} />
                <p className="text-[10px] font-semibold text-[#5C5854] uppercase tracking-[0.15em]">Processing Module</p>
                <div className="ml-auto text-[10px] font-mono text-[#36322E]">
                  {loading ? "RUNNING" : "STANDBY"}
                </div>
              </div>

              <div className="p-4 space-y-4">

                {/* ── Tab selector: console-style ── */}
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-[#080808] border border-[#1E1B18]">
                  {([
                    { value: "separate", label: "Stem Separation", icon: "⊗", activeColor: "text-[#A78BFA] border-[#A78BFA]/60 bg-[#A78BFA]/10 shadow-[0_0_12px_rgba(167,139,250,0.15)]" },
                    { value: "eq-ai",    label: "Auto EQ (AI)",    icon: "⟿", activeColor: "text-[#22D3EE] border-[#22D3EE]/60 bg-[#22D3EE]/10 shadow-[0_0_12px_rgba(34,211,238,0.15)]" },
                    { value: "compressor",label: "Compressor",     icon: "◉", activeColor: "text-[#E5A93D] border-[#E5A93D]/60 bg-[#E5A93D]/10 shadow-[0_0_12px_rgba(229,169,61,0.15)]" },
                    { value: "pitch",    label: "Pitch Shift",     icon: "♯", activeColor: "text-[#34D399] border-[#34D399]/60 bg-[#34D399]/10 shadow-[0_0_12px_rgba(52,211,153,0.15)]" },
                  ] as const).map((item) => (
                    <button
                      key={item.value}
                      onClick={() => setAction(item.value)}
                      disabled={loading}
                      className={`relative flex flex-col items-center gap-0.5 rounded-md px-2 py-2.5 text-xs font-medium border transition-all duration-200 cursor-pointer ${
                        action === item.value
                          ? item.activeColor
                          : "text-[#444444] border-transparent bg-transparent hover:text-[#888888] hover:bg-[#111111]"
                      } disabled:opacity-40 disabled:cursor-not-allowed`}
                    >
                      <span className={`text-base leading-none transition-all duration-200 ${
                        action === item.value ? "opacity-100" : "opacity-40"
                      }`}>{item.icon}</span>
                      <span className="leading-tight text-center">{item.label}</span>
                    </button>
                  ))}
                </div>

                {/* ── Panel ตั้งค่าแต่ละ Module ── */}
                {/* Stem Separation: ไม่มี settings เพิ่มเติม แสดง info card */}
                {action === "separate" && (
                  <div className="rounded-lg border border-[#A78BFA]/20 bg-[#A78BFA]/5 p-3 space-y-2">
                    <p className="text-xs text-[#A78BFA] font-medium">Stem Separation</p>
                    <p className="text-[11px] text-[#666666] leading-relaxed">
                      แยกไฟล์เสียงออกเป็น 4 แทร็กอิสระ — Vocals, Drums, Bass, Other — พร้อมเล่นและ Mix ได้ทันที
                    </p>
                    <div className="grid grid-cols-4 gap-1 pt-1">
                      {["Vocals","Drums","Bass","Other"].map((s) => (
                        <div key={s} className="rounded bg-[#A78BFA]/10 border border-[#A78BFA]/20 py-1 text-center text-[10px] text-[#A78BFA]/80">{s}</div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Genre Profile (เฉพาะ EQ / Compressor) */}
                {(action === "eq-ai" || action === "compressor") && (
                  <div>
                    <label className="block text-[10px] font-semibold text-[#5C5854] uppercase tracking-[0.12em] mb-1.5">Genre Profile</label>
                    <select
                      value={genre}
                      onChange={(e) => setGenre(e.target.value)}
                      className={`w-full rounded-lg bg-[#080808] border p-2.5 text-[#C8C8C8] text-sm focus:outline-none transition ${
                        action === "eq-ai"
                          ? "border-[#22D3EE]/30 focus:border-[#22D3EE]/70"
                          : "border-[#E5A93D]/30 focus:border-[#F97316]/70"
                      }`}
                      disabled={loading}
                    >
                      <option value="pop">Pop</option>
                      <option value="rock">Rock</option>
                      <option value="trap">Trap</option>
                      <option value="country">Country</option>
                      <option value="soul">Soul</option>
                    </select>
                  </div>
                )}

                {/* Auto EQ Settings */}
                {action === "eq-ai" && (
                  <div className="rounded-lg border border-[#22D3EE]/20 bg-[#22D3EE]/5 p-3">
                    <AutoEqSettings
                      autoEqModel={autoEqModel}
                      setAutoEqModel={setAutoEqModel}
                      deltaClampDb={deltaClampDb}
                      setDeltaClampDb={setDeltaClampDb}
                      loading={loading}
                      modelOptions={AUTO_EQ_MODEL_OPTIONS}
                      minDeltaClamp={AUTO_EQ_DELTA_CLAMP_MIN}
                      maxDeltaClamp={AUTO_EQ_DELTA_CLAMP_MAX}
                      defaultDeltaClamp={AUTO_EQ_DELTA_CLAMP_DEFAULT}
                      isValid={isEqDeltaClampValid}
                      warningText={eqDeltaClampWarning}
                    />
                  </div>
                )}

                {/* Compressor Settings */}
                {action === "compressor" && (
                  <div className="rounded-lg border border-[#E5A93D]/20 bg-[#E5A93D]/5 p-3">
                    <CompressorSettings
                      params={compParams}
                      onChange={patchCompParams}
                      loading={loading}
                    />
                  </div>
                )}

                {/* Pitch Shift Settings */}
                {action === "pitch" && (
                  <div className="rounded-lg border border-[#34D399]/20 bg-[#34D399]/5 p-3 space-y-3">
                    <PitchShiftSettings
                      pitchSteps={pitchSteps}
                      setPitchSteps={setPitchSteps}
                      loading={loading}
                    />
                  </div>
                )}

                {/* ── Trimming ── */}
                <div className="rounded-lg border border-[#1E1B18] bg-[#080808] p-3 space-y-3">
                  <label className="flex items-center gap-2.5 cursor-pointer select-none group">
                    {/* Custom checkbox */}
                    <span className={`w-4 h-4 rounded flex-shrink-0 border transition-all duration-150 flex items-center justify-center ${
                      isTrimming ? "bg-[#E5A93D] border-[#E5A93D]" : "border-[#36322E] bg-transparent group-hover:border-[#5C5854]"
                    }`}
                      onClick={() => !loading && setIsTrimming(!isTrimming)}
                    >
                      {isTrimming && (
                        <svg className="w-2.5 h-2.5 text-[#0D0B0A]" viewBox="0 0 10 10" fill="none">
                          <path d="M1.5 5L4 7.5L8.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </span>
                    <input type="checkbox" checked={isTrimming} onChange={(e) => setIsTrimming(e.target.checked)} disabled={loading} className="sr-only" />
                    <span className="text-xs font-medium text-[#888888] group-hover:text-[#AAAAAA] transition-colors">Trim — ตัดช่วงเวลาก่อนประมวลผล</span>
                  </label>
                  {isTrimming && (
                    <div className="grid grid-cols-2 gap-2 pl-6">
                      <div>
                        <label className="block text-[10px] text-[#5C5854] mb-1">เริ่ม (วินาที)</label>
                        <input
                          type="number" min="0" value={trimStart}
                          onChange={(e) => setTrimStart(e.target.value)} disabled={loading}
                          className="w-full rounded-md bg-[#111111] border border-[#2C2824] px-2.5 py-1.5 text-sm text-[#E0E0E0] focus:border-[#F97316]/50 focus:outline-none transition"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-[#5C5854] mb-1">สิ้นสุด (วินาที)</label>
                        <input
                          type="number" min="1" value={trimEnd}
                          onChange={(e) => setTrimEnd(e.target.value)} disabled={loading}
                          className="w-full rounded-md bg-[#111111] border border-[#2C2824] px-2.5 py-1.5 text-sm text-[#E0E0E0] focus:border-[#F97316]/50 focus:outline-none transition"
                        />
                      </div>
                    </div>
                  )}
                </div>


                {/* ── Execute Button ── */}
                {(() => {
                  // สีและ shadow ของปุ่มเปลี่ยนตาม module ที่เลือก
                  const moduleStyles: Record<AudioAction, { base: string; glow: string; label: string }> = {
                    separate:   { base: "from-[#7C3AED] to-[#A78BFA]", glow: "rgba(167,139,250,0.35)", label: "Run Stem Separation" },
                    "eq-ai":    { base: "from-[#0891B2] to-[#22D3EE]", glow: "rgba(34,211,238,0.35)",  label: "Apply Auto EQ" },
                    compressor: { base: "from-[#B45309] to-[#E5A93D]", glow: "rgba(229,169,61,0.35)",  label: "Apply Compressor" },
                    pitch:      { base: "from-[#059669] to-[#34D399]", glow: "rgba(52,211,153,0.35)",  label: "Shift Pitch" },
                  };
                  const moduleStyle = moduleStyles[action] ?? moduleStyles.compressor;
                  const isDisabled = loading || (action === "eq-ai" && !isEqDeltaClampValid);
                  return (
                    <button
                      onClick={handleUpload}
                      disabled={isDisabled}
                      data-testid="process-button"
                      className={`relative w-full rounded-lg py-3.5 text-sm font-bold tracking-widest uppercase overflow-hidden transition-all duration-300 ${
                        isDisabled
                          ? "bg-[#1E1B18] text-[#36322E] cursor-not-allowed border border-[#222222]"
                          : `bg-gradient-to-r ${moduleStyle.base} text-white cursor-pointer`
                      }`}
                      style={!isDisabled ? { boxShadow: loading ? `0 0 20px ${moduleStyle.glow}` : `0 0 10px ${moduleStyle.glow}` } : {}}
                    >
                      {/* shimmer overlay เมื่อ processing */}
                      {loading && (
                        <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-[shimmer_1.5s_linear_infinite] bg-[length:200%_100%]" />
                      )}
                      <span className="relative flex items-center justify-center gap-2">
                        {loading ? (
                          <>
                            <div className="flex items-end gap-[3px] h-3.5 w-5 pb-[1px]">
                              <div className="w-[3px] h-full bg-white rounded-full origin-bottom animate-audiowave" style={{ animationDelay: "0.1s" }} />
                              <div className="w-[3px] h-full bg-white rounded-full origin-bottom animate-audiowave" style={{ animationDelay: "0.3s" }} />
                              <div className="w-[3px] h-full bg-white rounded-full origin-bottom animate-audiowave" style={{ animationDelay: "0.5s" }} />
                              <div className="w-[3px] h-full bg-white rounded-full origin-bottom animate-audiowave" style={{ animationDelay: "0.2s" }} />
                            </div>
                            Processing…
                          </>
                        ) : moduleStyle.label}
                      </span>
                    </button>
                  );
                })()}

                {/* ── Progress + Status ── */}
                <div className="space-y-2">
                  <div className="flex flex-col gap-1.5 w-full">
                    {/* Indeterminate progress bar — เคลื่อนที่ไปมาไม่แสดง % */}
                    <div className="w-full h-1.5 overflow-hidden rounded-full bg-[#111111] border border-[#1E1B18]">
                      <div 
                        className={`h-full rounded-full transition-all ${
                          action === "separate" ? "bg-gradient-to-r from-[#A78BFA] via-[#C084FC] to-[#A78BFA]" 
                          : action === "eq-ai" ? "bg-gradient-to-r from-[#22D3EE] via-[#67e8f9] to-[#22D3EE]"
                          : action === "compressor" ? "bg-gradient-to-r from-[#E5A93D] via-[#FBBF24] to-[#E5A93D]"
                          : "bg-gradient-to-r from-[#34D399] via-[#6EE7B7] to-[#34D399]"
                        } ${loading ? "bg-[length:200%_auto] animate-shimmer" : ""}`}
                        style={{ width: loading ? "60%" : "100%" }}
                      />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-[11px] text-[#5C5854]">
                        {statusText || (loading ? "กำลังประมวลผล..." : "พร้อมใช้งาน")}
                      </span>
                      {/* ไม่แสดง % ปลอมอีกต่อไป */}
                    </div>
                    {processingTime && (
                      <div className="text-[11px] text-[#444444] font-mono">⏱ {processingTime}</div>
                    )}
                  </div>
                </div>

                {/* ── Error / Success ── */}
                {errorMessage && (
                  <div className="flex items-start gap-2 rounded-lg border border-red-900/40 bg-red-950/30 px-3 py-2.5">
                    <svg className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 10.5a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5zm.75-3.25a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0v3z"/>
                    </svg>
                    <span className="text-xs text-red-400 leading-snug">{errorMessage}</span>
                  </div>
                )}
                {!loading && successMessage && (
                  <div className={`flex items-start gap-2 rounded-lg border px-3 py-2.5 ${
                    action === "separate" ? "border-[#A78BFA]/30 bg-[#A78BFA]/10"
                    : action === "eq-ai" ? "border-[#22D3EE]/30 bg-[#22D3EE]/10"
                    : action === "compressor" ? "border-[#E5A93D]/30 bg-[#E5A93D]/10"
                    : "border-[#34D399]/30 bg-[#34D399]/10"
                  }`}>
                    <svg className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${
                      action === "separate" ? "text-[#A78BFA]"
                      : action === "eq-ai" ? "text-[#22D3EE]"
                      : action === "compressor" ? "text-[#E5A93D]"
                      : "text-[#34D399]"
                    }`} viewBox="0 0 16 16" fill="currentColor">
                      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm3.28 5.28-4 4a.75.75 0 0 1-1.06 0l-2-2a.75.75 0 1 1 1.06-1.06l1.47 1.47 3.47-3.47a.75.75 0 0 1 1.06 1.06z"/>
                    </svg>
                    <span className={`text-xs leading-snug ${
                      action === "separate" ? "text-[#A78BFA]"
                      : action === "eq-ai" ? "text-[#22D3EE]"
                      : action === "compressor" ? "text-[#E5A93D]"
                      : "text-[#34D399]"
                    }`}>{successMessage}</span>
                  </div>
                )}

              </div>
            </div>
          </div>

          <div className="space-y-4 lg:col-span-8">
            {loading && (
              <>
                <AudioAnalysisSkeleton />
                {action === "separate" ? <StemMixerSkeleton /> : <SinglePlayerSkeleton />}
              </>
            )}

            {!loading && (analysis || fileId || downloadUrl) && (
              <div className="animate-fade-in-up space-y-4">
                {analysis && <AudioAnalysis data={analysis} />}

                {fileId && (
                  <div>
                    <MultiStemLivePlayer fileId={fileId} />
                  </div>
                )}

                {zipUrl && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {fileId && (
                      <button
                        onClick={() => setIsExportModalOpen(true)}
                        className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0D0B0A] shadow-[0_4px_15px_rgba(229,169,61,0.2)] transition-all hover:shadow-[0_6px_25px_rgba(229,169,61,0.35)] hover:from-[#F3C05D] hover:to-[#E5A93D]"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        Export & Download
                      </button>
                    )}
                    {fileId && (
                      <button
                        onClick={handleKaraokeDownload}
                        className="cursor-pointer flex w-full items-center justify-center gap-2 rounded-xl border border-[#2C2824] bg-[#161412] px-4 py-3.5 font-semibold text-white shadow-[0_4px_15px_rgba(0,0,0,0.2)] transition-all hover:border-[#E5A93D]/50 hover:text-[#E5A93D] hover:bg-[#1E1B18]"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 opacity-70" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.938l3-8V5a1 1 0 00-1-1H4a1 1 0 00-1 1v1.938l3 8V17a3 3 0 006 0v-2.062z" clipRule="evenodd" />
                        </svg>
                        Download Karaoke
                      </button>
                    )}
                  </div>
                )}

                {downloadUrl && downloadFileName && !downloadFileName.endsWith(".zip") && (
                  <div className="rounded-2xl border border-[#222] bg-[#0D0B0A] p-5 space-y-5 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
                    <div>
                      <WaveformPlayer audioUrl={downloadUrl} />
                    </div>
                    <button
                      onClick={() => setIsSingleExportModalOpen(true)}
                      className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0D0B0A] shadow-[0_4px_15px_rgba(229,169,61,0.2)] transition-all hover:shadow-[0_6px_25px_rgba(229,169,61,0.35)] hover:from-[#F3C05D] hover:to-[#E5A93D]"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                      Export & Download
                    </button>
                  </div>
                )}
              </div>
            )}

            <ExportMasterModal 
              isOpen={isExportModalOpen} 
              onClose={() => setIsExportModalOpen(false)} 
              onExport={handleExport}
              isExporting={isExporting}
            />

            {/* M11: key บังคับ remount ทุกครั้งที่เปิด modal หรือ format เปลี่ยน
                (กัน SingleExportModal แสดง radio format ค้างจากครั้งก่อน) */}
            <SingleExportModal
              key={`${exportFormat}-${isSingleExportModalOpen}`}
              isOpen={isSingleExportModalOpen}
              onClose={() => setIsSingleExportModalOpen(false)}
              onExport={handleSingleExport}
              isExporting={isExporting}
              currentFormat={exportFormat}
            />
          </div>
        </>
      )}
    </div>
  );
}

export default UploadBox;
