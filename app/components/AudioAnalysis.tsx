"use client";
import React from "react";

interface AnalysisProps {
  data: {
    tempo: number;
    key: string;
    pitch: string | null;
  };
}

// การ์ดแสดงผลวิเคราะห์เสียงที่ backend ส่งกลับมา เช่น tempo, key และ pitch
const AudioAnalysis: React.FC<AnalysisProps> = ({ data }) => {
  return (
    <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 shadow-lg">
      <div className="flex items-center gap-2 text-[#EDE9FE]">
        <span className="text-lg">การวิเคราะห์เสียง</span>
        <h3 className="text-lg font-semibold">Audio Analysis</h3>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-4 text-center text-[#EDE9FE]">
        <div className="rounded-lg bg-[#1A1A1A] p-3 border border-[#2A2A2A]">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Tempo</p>
          <p className="text-xl font-semibold">{Math.round(data.tempo)} BPM</p>
        </div>
        <div className="rounded-lg bg-[#1A1A1A] p-3 border border-[#2A2A2A]">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Key</p>
          <p className="text-xl font-semibold">{data.key}</p>
        </div>
        <div className="rounded-lg bg-[#1A1A1A] p-3 border border-[#2A2A2A]">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Pitch</p>
          <p className="text-xl font-semibold">{data.pitch || "-"}</p>
        </div>
      </div>
    </div>
  );
};

export default AudioAnalysis;
