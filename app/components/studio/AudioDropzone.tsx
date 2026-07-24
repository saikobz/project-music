"use client";

import React, { useRef } from "react";

interface AudioDropzoneProps {
  file: File | null;
  isDragging: boolean;
  isTrimming: boolean;
  trimStart: string;
  trimEnd: string;
  onDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: (e: React.DragEvent<HTMLDivElement>) => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onToggleTrim: () => void;
  onTrimStartChange: (v: string) => void;
  onTrimEndChange: (v: string) => void;
}

export function AudioDropzone({
  file,
  isDragging,
  isTrimming,
  trimStart,
  trimEnd,
  onDragOver,
  onDragLeave,
  onDrop,
  onFileChange,
  onToggleTrim,
  onTrimStartChange,
  onTrimEndChange,
}: AudioDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
        isDragging
          ? "border-[#E5A93D] bg-[#E5A93D]/10"
          : file
          ? "border-emerald-500/50 bg-emerald-500/5"
          : "border-[#2A2A2A] hover:border-[#3A3A3A] bg-[#121212]"
      }`}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".wav,audio/wav,audio/x-wav"
        onChange={onFileChange}
        className="hidden"
      />

      <div className="flex flex-col items-center justify-center space-y-3">
        <div className="w-12 h-12 rounded-full bg-[#1E1E1E] flex items-center justify-center border border-[#2A2A2A]">
          <svg className="w-6 h-6 text-[#E5A93D]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>

        {file ? (
          <div>
            <p className="text-sm font-semibold text-emerald-400">เลือกไฟล์สำเร็จ: {file.name}</p>
            <p className="text-xs text-slate-400 mt-1">{(file.size / (1024 * 1024)).toFixed(2)} MB • WAV Format</p>
          </div>
        ) : (
          <div>
            <p className="text-sm font-medium text-slate-200">ลากไฟล์ WAV มาวางที่นี่ หรือคลิกเพื่ออัปโหลด</p>
            <p className="text-xs text-slate-400 mt-1">รองรับไฟล์ .WAV ขนาดสูงสุด 100MB</p>
          </div>
        )}

        <div className="pt-2 flex items-center gap-3 text-xs" onClick={(e) => e.stopPropagation()}>
          <label className="flex items-center gap-2 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={isTrimming}
              onChange={onToggleTrim}
              className="accent-[#E5A93D] rounded"
            />
            <span>ตัดช่วงเวลาเสียง (Trim)</span>
          </label>
          {isTrimming && (
            <div className="flex items-center gap-2 bg-[#1A1A1A] p-1.5 rounded-lg border border-[#2A2A2A]">
              <input
                type="number"
                value={trimStart}
                onChange={(e) => onTrimStartChange(e.target.value)}
                placeholder="เริ่ม (วิ)"
                className="w-16 bg-[#121212] border border-[#2A2A2A] rounded px-2 py-0.5 text-center text-xs"
              />
              <span className="text-slate-500">-</span>
              <input
                type="number"
                value={trimEnd}
                onChange={(e) => onTrimEndChange(e.target.value)}
                placeholder="จบ (วิ)"
                className="w-16 bg-[#121212] border border-[#2A2A2A] rounded px-2 py-0.5 text-center text-xs"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
