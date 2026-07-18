"use client";

import React, { useState } from "react";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onExport: (targetLufs: number, selectedStems: string[]) => void;
  isExporting: boolean;
};

const STEM_OPTIONS = [
  { value: "mix", label: "Stereo Mix (Mastered)" },
  { value: "vocals", label: "Vocals (เสียงร้อง)" },
  { value: "drums", label: "Drums (กลอง)" },
  { value: "bass", label: "Bass (เบส)" },
  { value: "other", label: "Other (เครื่องดนตรีอื่น ๆ)" },
];

export default function ExportMasterModal({ isOpen, onClose, onExport, isExporting }: Props) {
  const [targetLufs, setTargetLufs] = useState<number>(-14.0);
  const [selectedStems, setSelectedStems] = useState<string[]>(["mix", "vocals", "drums", "bass", "other"]);

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
      alert("กรุณาเลือกอย่างน้อย 1 รายการเพื่อ Export");
      return;
    }
    onExport(targetLufs, selectedStems);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-[450px] max-w-full rounded-2xl border border-[#2A2A2A] bg-[#121212] p-6 shadow-2xl">
        <div className="mb-4 text-xl font-bold text-white">Export & Mastering</div>
        
        {/* เลือกระดับความดัง */}
        <div className="mb-4">
          <div className="mb-2 text-sm font-semibold text-gray-300">1. เลือกระดับความดัง (LUFS Target)</div>
          <div className="space-y-2">
            <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-[#2A2A2A] bg-[#1A1A1A] px-3 py-2 hover:border-[#E5A93D]">
              <input
                type="radio"
                name="lufs"
                value={-14}
                checked={targetLufs === -14.0}
                onChange={() => setTargetLufs(-14.0)}
                className="accent-[#E5A93D]"
              />
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-white">Spotify / YouTube (-14 LUFS)</span>
              </div>
            </label>
            <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-[#2A2A2A] bg-[#1A1A1A] px-3 py-2 hover:border-[#E5A93D]">
              <input
                type="radio"
                name="lufs"
                value={-16}
                checked={targetLufs === -16.0}
                onChange={() => setTargetLufs(-16.0)}
                className="accent-[#E5A93D]"
              />
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-white">Apple Music (-16 LUFS)</span>
              </div>
            </label>
            <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-[#2A2A2A] bg-[#1A1A1A] px-3 py-2 hover:border-[#E5A93D]">
              <input
                type="radio"
                name="lufs"
                value={-9}
                checked={targetLufs === -9.0}
                onChange={() => setTargetLufs(-9.0)}
                className="accent-[#E5A93D]"
              />
              <div className="flex flex-col">
                <span className="text-sm font-semibold text-white">CD / Club / Loud (-9 LUFS)</span>
              </div>
            </label>
          </div>
        </div>

        {/* เลือก Stems ที่จะ Export */}
        <div className="mb-6">
          <div className="mb-2 text-sm font-semibold text-gray-300">2. เลือกแทร็กที่ต้องการส่งออก (Export Stems)</div>
          <div className="rounded-lg border border-[#2A2A2A] bg-[#1A1A1A] p-3 space-y-2">
            {STEM_OPTIONS.map((option) => (
              <label key={option.value} className="flex cursor-pointer items-center gap-3 text-sm text-gray-300 hover:text-white">
                <input
                  type="checkbox"
                  checked={selectedStems.includes(option.value)}
                  onChange={() => handleToggleStem(option.value)}
                  className="rounded border-[#2A2A2A] accent-[#E5A93D] h-4 w-4"
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-semibold text-gray-400 hover:text-white"
            disabled={isExporting}
          >
            ยกเลิก
          </button>
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="flex items-center justify-center rounded-lg bg-[#E5A93D] px-5 py-2.5 text-sm font-semibold text-black hover:bg-[#F3C05D] disabled:opacity-50"
          >
            {isExporting ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-black border-t-transparent"></span>
                กำลังประมวลผล...
              </span>
            ) : (
              "Export & Download"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
