"use client";

import React, { useState } from "react";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  onExport: (format: string) => void;
  isExporting: boolean;
  currentFormat: string;
};

export default function SingleExportModal({ isOpen, onClose, onExport, isExporting, currentFormat }: Props) {
  const [format, setFormat] = useState<string>(currentFormat);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#050505]/80 backdrop-blur-md transition-opacity p-4">
      <div className="w-full max-w-[400px] rounded-2xl border border-[#2A2A2A] bg-[#0A0A0A] p-6 shadow-2xl">
        <div className="mb-6 flex items-center justify-between">
          <h2 className="text-xl font-bold tracking-tight text-white">Export & Download</h2>
          <button onClick={onClose} className="text-[#8E8E8E] hover:text-white transition-colors" disabled={isExporting}>
             <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
        
        {/* Step 1: Format */}
        <div className="mb-6">
          <div className="mb-3 text-xs font-medium uppercase tracking-widest text-[#8E8E8E]">1. Format</div>
          <div className="flex flex-col gap-2">
            <label className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 transition-colors ${format === "wav" ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
              <input
                type="radio"
                name="single_format"
                value="wav"
                checked={format === "wav"}
                onChange={() => setFormat("wav")}
                className="accent-[#E5A93D]"
              />
              <span className="text-sm font-medium text-white">WAV <span className="text-xs text-[#8E8E8E] font-normal">(Lossless)</span></span>
            </label>
            <label className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 transition-colors ${format === "mp3" ? "border-[#E5A93D] bg-[#E5A93D]/5" : "border-[#2A2A2A] bg-[#121212] hover:border-[#444444]"}`}>
              <input
                type="radio"
                name="single_format"
                value="mp3"
                checked={format === "mp3"}
                onChange={() => setFormat("mp3")}
                className="accent-[#E5A93D]"
              />
              <span className="text-sm font-medium text-white">MP3 <span className="text-xs text-[#8E8E8E] font-normal">(320kbps)</span></span>
            </label>
          </div>
        </div>

        {/* Action Button */}
        <button
          onClick={() => onExport(format)}
          disabled={isExporting}
          className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0A0A0A] transition-all hover:shadow-[0_0_20px_rgba(229,169,61,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isExporting ? (
            <>
              <svg className="h-5 w-5 animate-spin text-[#0A0A0A]" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
              Export & Download
            </>
          )}
        </button>
      </div>
    </div>
  );
}
