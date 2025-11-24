"use client";
import React from "react";

interface AnalysisProps {
  data: {
    tempo: number;
    key: string;
    pitch: string | null;
  };
}

const AudioAnalysis: React.FC<AnalysisProps> = ({ data }) => {
  return (
    <div className="rounded-2xl border border-[#5B21B6]/40 bg-gradient-to-br from-[#111827] via-[#1F1B34] to-[#312E81] p-5 shadow-[0_20px_40px_rgba(17,24,39,0.45)]">
      <div className="flex items-center gap-2 text-[#EDE9FE]">
        <span className="text-lg">การวิเคราะห์เสียง</span>
        <h3 className="text-lg font-semibold">Audio Analysis</h3>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-4 text-center text-[#EDE9FE]">
        <div className="rounded-xl bg-[#5B21B6]/20 p-3 border border-[#5B21B6]/30">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Tempo</p>
          <p className="text-xl font-semibold">{Math.round(data.tempo)} BPM</p>
        </div>
        <div className="rounded-xl bg-[#5B21B6]/20 p-3 border border-[#5B21B6]/30">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Key</p>
          <p className="text-xl font-semibold">{data.key}</p>
        </div>
        <div className="rounded-xl bg-[#5B21B6]/20 p-3 border border-[#5B21B6]/30">
          <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Pitch</p>
          <p className="text-xl font-semibold">{data.pitch || "-"}</p>
        </div>
      </div>
    </div>
  );
};

export default AudioAnalysis;
