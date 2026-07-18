"use client";

import React, { useState } from "react";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onExport: (exportType: string, exportFormat: string, targetLufs: number, selectedStems: string[]) => void;
  isExporting: boolean;
};

const STEM_OPTIONS = [
  { value: "vocals", label: "Vocals (เสียงร้อง)" },
  { value: "drums", label: "Drums (กลอง)" },
  { value: "bass", label: "Bass (เบส)" },
  { value: "other", label: "Other (อื่นๆ)" },
];

export default function ExportMasterModal({ isOpen, onClose, onExport, isExporting }: Props) {
  const [exportType, setExportType] = useState<string>("mix"); // "mix" | "stems"
  const [exportFormat, setExportFormat] = useState<string>("wav"); // "wav" | "mp3"
  const [targetLufs, setTargetLufs] = useState<number>(-14.0);
  const [selectedStems, setSelectedStems] = useState<string[]>(["vocals", "drums", "bass", "other"]);

  if (!isOpen) return null;

  const handleToggleStem = (value: string) => {
    if (selectedStems.includes(value)) {
      setSelectedStems(selectedStems.filter((item) => item !== value));
    } else {
      setSelectedStems([...selectedStems, value]);
    }
  };

  const handleExport = () => {
    if (selectedStems.length === 0) {
      alert("กรุณาเลือกอย่างน้อย 1 แทร็กเพื่อ Export");
      return;
    }
    onExport(exportType, exportFormat, targetLufs, selectedStems);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#050505]/80 backdrop-blur-md transition-opacity p-4">
      <div className="w-full max-w-[540px] rounded-2xl border border-[#2A2A2A] bg-[#0A0A0A] p-6 shadow-2xl">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-bold tracking-tight text-white">Export & Download</h2>
          <button onClick={onClose} className="text-[#8E8E8E] hover:text-white transition-colors" disabled={isExporting}>
             <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
        
        {/* Step 1: Export Type */}
        <div className="mb-6">
          <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E]">1. Output Type</div>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setExportType("mix")}
              className={`rounded-xl border p-4 text-left transition-all ${
                exportType === "mix"
                  ? "border-[#E5A93D] bg-[#E5A93D]/10 shadow-[0_0_15px_rgba(229,169,61,0.1)]"
                  : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"
              }`}
            >
              <div className={`mb-1 font-semibold ${exportType === "mix" ? "text-[#E5A93D]" : "text-white"}`}>Mixed Audio</div>
              <div className="text-xs text-[#8E8E8E]">รวมทุกแทร็กที่เลือกเป็นไฟล์เดียว</div>
            </button>
            <button
              onClick={() => setExportType("stems")}
              className={`rounded-xl border p-4 text-left transition-all ${
                exportType === "stems"
                  ? "border-[#E5A93D] bg-[#E5A93D]/10 shadow-[0_0_15px_rgba(229,169,61,0.1)]"
                  : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"
              }`}
            >
              <div className={`mb-1 font-semibold ${exportType === "stems" ? "text-[#E5A93D]" : "text-white"}`}>Separate Stems</div>
              <div className="text-xs text-[#8E8E8E]">แยกแทร็กที่เลือก (.zip)</div>
            </button>
          </div>
        </div>

        {/* Step 2: Format & Stems */}
        <div className="mb-6 grid grid-cols-2 gap-6">
          <div>
            <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E]">2. Format</div>
            <div className="flex flex-col gap-2">
              <label className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 transition-colors ${exportFormat === "wav" ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
                <input
                  type="radio"
                  name="format"
                  value="wav"
                  checked={exportFormat === "wav"}
                  onChange={() => setExportFormat("wav")}
                  className="accent-[#E5A93D]"
                />
                <span className="text-sm font-medium text-white">WAV <span className="text-xs text-[#8E8E8E] font-normal">(Lossless)</span></span>
              </label>
              <label className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 transition-colors ${exportFormat === "mp3" ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
                <input
                  type="radio"
                  name="format"
                  value="mp3"
                  checked={exportFormat === "mp3"}
                  onChange={() => setExportFormat("mp3")}
                  className="accent-[#E5A93D]"
                />
                <span className="text-sm font-medium text-white">MP3 <span className="text-xs text-[#8E8E8E] font-normal">(320kbps)</span></span>
              </label>
            </div>
          </div>
          
          <div>
            <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E]">3. Include Stems</div>
            <div className="flex flex-col gap-2 rounded-lg border border-[#2A2A2A] bg-[#121212] p-3">
              {STEM_OPTIONS.map((option) => (
                <label key={option.value} className="flex cursor-pointer items-center gap-3 text-sm text-gray-300 transition-colors hover:text-white">
                  <input
                    type="checkbox"
                    checked={selectedStems.includes(option.value)}
                    onChange={() => handleToggleStem(option.value)}
                    className="h-4 w-4 rounded border-[#2A2A2A] accent-[#E5A93D]"
                  />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Step 3: LUFS (Only for Mix) */}
        {exportType === "mix" ? (
          <div className="mb-8">
            <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E]">4. Mastering Level (LUFS)</div>
            <div className="grid grid-cols-3 gap-3">
              <label className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border p-3 transition-colors ${targetLufs === -16.0 ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
                <input type="radio" name="lufs" value={-16} checked={targetLufs === -16.0} onChange={() => setTargetLufs(-16.0)} className="mb-2 accent-[#E5A93D]" />
                <span className="text-sm font-bold text-white">-16</span>
                <span className="text-[10px] text-[#8E8E8E] text-center mt-1">Apple Music</span>
              </label>
              <label className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border p-3 transition-colors ${targetLufs === -14.0 ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
                <input type="radio" name="lufs" value={-14} checked={targetLufs === -14.0} onChange={() => setTargetLufs(-14.0)} className="mb-2 accent-[#E5A93D]" />
                <span className="text-sm font-bold text-white">-14</span>
                <span className="text-[10px] text-[#8E8E8E] text-center mt-1">Spotify / YT</span>
              </label>
              <label className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border p-3 transition-colors ${targetLufs === -9.0 ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
                <input type="radio" name="lufs" value={-9} checked={targetLufs === -9.0} onChange={() => setTargetLufs(-9.0)} className="mb-2 accent-[#E5A93D]" />
                <span className="text-sm font-bold text-white">-9</span>
                <span className="text-[10px] text-[#8E8E8E] text-center mt-1">CD / Loud</span>
              </label>
            </div>
          </div>
        ) : (
          <div className="mb-8">
            <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E] opacity-50">4. Mastering Level (LUFS)</div>
            <div className="rounded-lg border border-[#2A2A2A] bg-[#121212]/50 p-4 text-center text-xs text-[#8E8E8E]">
              Mastering level is only available when exporting a Mixed Audio.
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3">
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="flex-1 rounded-xl bg-[#E5A93D] py-3.5 text-sm font-bold uppercase tracking-wider text-[#0A0A0A] transition-all hover:bg-[#F3C05D] disabled:opacity-50 shadow-[0_0_15px_rgba(229,169,61,0.2)]"
          >
            {isExporting ? (
              <span className="flex items-center justify-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-[#0A0A0A] border-t-transparent"></span>
                Processing...
              </span>
            ) : (
              "Download"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

