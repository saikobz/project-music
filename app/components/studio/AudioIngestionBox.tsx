// app/components/studio/AudioIngestionBox.tsx
"use client";

import React, { useRef } from "react";
import { Upload, FileAudio, Trash2, Scissors } from "lucide-react";

interface AudioIngestionBoxProps {
  file: File | null;
  isDragging: boolean;
  onFileSelect: (file: File | null) => void;
  onDragOver: (e: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: () => void;
  onDrop: (e: React.DragEvent<HTMLDivElement>) => void;
  isTrimming: boolean;
  setIsTrimming: (trimming: boolean) => void;
  trimStart: string;
  setTrimStart: (val: string) => void;
  trimEnd: string;
  setTrimEnd: (val: string) => void;
  disabled?: boolean;
}

export function AudioIngestionBox({
  file,
  isDragging,
  onFileSelect,
  onDragOver,
  onDragLeave,
  onDrop,
  isTrimming,
  setIsTrimming,
  trimStart,
  setTrimStart,
  trimEnd,
  setTrimEnd,
  disabled = false,
}: AudioIngestionBoxProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="space-y-4">
      {/* File Drop Area */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-[#E5A93D] bg-[#E5A93D]/10"
            : file
            ? "border-[#E5A93D]/50 bg-[#E5A93D]/5"
            : "border-[#2A2A2A] bg-[#121212] hover:border-[#E5A93D]/40 hover:bg-[#1A1A1A]"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,audio/wav"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              onFileSelect(e.target.files[0]);
            }
          }}
        />

        {file ? (
          <div className="flex items-center justify-between p-2">
            <div className="flex items-center gap-3 text-left">
              <div className="p-3 rounded-xl bg-[#E5A93D]/20 text-[#E5A93D]">
                <FileAudio className="h-6 w-6" />
              </div>
              <div>
                <p className="text-sm font-semibold text-white truncate max-w-[280px]">
                  {file.name}
                </p>
                <p className="text-xs text-[#888888]">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB &bull; WAV
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onFileSelect(null);
              }}
              className="p-2 text-[#888888] hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
            >
              <Trash2 className="h-5 w-5" />
            </button>
          </div>
        ) : (
          <div className="space-y-3 py-4">
            <div className="mx-auto w-12 h-12 rounded-full bg-[#1A1A1A] border border-[#2A2A2A] flex items-center justify-center text-[#E5A93D]">
              <Upload className="h-6 w-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-white">
                ลากไฟล์มาวางที่นี่ หรือ <span className="text-[#E5A93D]">คลิกเพื่อเลือกไฟล์</span>
              </p>
              <p className="text-xs text-[#888888] mt-1">
                รองรับไฟล์ WAV ขนาดสูงสุดไม่เกิน 100MB
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Trimming Control Toggle */}
      {file && (
        <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Scissors className="h-4 w-4 text-[#E5A93D]" />
              <span className="text-xs font-semibold text-white">
                ตัดช่วงเวลาเสียง (Audio Trimming)
              </span>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={isTrimming}
                onChange={(e) => setIsTrimming(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-[#2A2A2A] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-[#E5A93D]"></div>
            </label>
          </div>

          {isTrimming && (
            <div className="grid grid-cols-2 gap-3 pt-2">
              <div>
                <label className="block text-[11px] text-[#888888] mb-1">เริ่มต้น (วินาที)</label>
                <input
                  type="number"
                  min="0"
                  value={trimStart}
                  onChange={(e) => setTrimStart(e.target.value)}
                  className="w-full bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg px-3 py-1.5 text-xs text-white focus:border-[#E5A93D] outline-none"
                />
              </div>
              <div>
                <label className="block text-[11px] text-[#888888] mb-1">สิ้นสุด (วินาที)</label>
                <input
                  type="number"
                  min="1"
                  value={trimEnd}
                  onChange={(e) => setTrimEnd(e.target.value)}
                  className="w-full bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg px-3 py-1.5 text-xs text-white focus:border-[#E5A93D] outline-none"
                />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
