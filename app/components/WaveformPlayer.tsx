"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

interface WaveformPlayerProps {
  audioUrl: string;
}

const WaveformPlayer: React.FC<WaveformPlayerProps> = ({ audioUrl }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const isDraggingRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    waveSurferRef.current?.destroy();

    waveSurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#C7D2FE",
      progressColor: "#5B21B6",
      cursorColor: "#22D3EE",
      height: 140,
      barGap: 2,
      barWidth: 2,
      responsive: true,
    });

    waveSurferRef.current.load(audioUrl);
    waveSurferRef.current.on("finish", () => setIsPlaying(false));
    waveSurferRef.current.on("error", (e) => {
      // ปล่อยผ่าน error ที่เกิดจากการ destroy/abort
      if ((e as any)?.name === "AbortError") return;
      console.error("WaveSurfer error", e);
    });

    return () => {
      waveSurferRef.current?.destroy();
    };
  }, [audioUrl]);

  const seekToPointer = (clientX: number) => {
    if (!waveSurferRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const progress = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    waveSurferRef.current.seekTo(progress);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = true;
    seekToPointer(event.clientX);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isDraggingRef.current) return;
    seekToPointer(event.clientX);
  };

  const handlePointerUp = () => {
    isDraggingRef.current = false;
  };

  const togglePlay = () => {
    waveSurferRef.current?.playPause();
    setIsPlaying((prev) => !prev);
  };

  const stopPlayback = () => {
    waveSurferRef.current?.stop();
    setIsPlaying(false);
  };

  return (
    <div className="space-y-3 rounded-2xl border border-[#7C3AED]/30 bg-[#1C162C] p-4 backdrop-blur">
      <div className="flex items-center justify-between text-sm font-semibold text-[#EDE9FE]">
        <span>ตัวเล่นไฟล์ที่ประมวลผลแล้ว (WAV)</span>
        <span className="text-[#A78BFA]">{isPlaying ? "กำลังเล่น" : "หยุดอยู่"}</span>
      </div>
      <div
        ref={containerRef}
        className="rounded-xl bg-[#0F0B1D] cursor-pointer"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      />
      <div className="flex items-center gap-3 justify-center">
        <button
          onClick={togglePlay}
          className="rounded-lg bg-[#7C3AED] px-4 py-2 font-semibold text-white hover:bg-[#A78BFA] cursor-pointer"
        >
          {isPlaying ? "หยุดชั่วคราว / เล่นต่อ" : "เล่น"}
        </button>
        <button
          onClick={stopPlayback}
          className="rounded-lg bg-[#F472B6] px-4 py-2 font-semibold text-white hover:bg-[#A78BFA] cursor-pointer"
        >
          หยุด
        </button>
      </div>
      <p className="text-center text-xs text-[#EDE9FE]/70">
        เลื่อนเพื่อ seek ไปยังตำแหน่งที่ต้องการบน waveform ได้ทันที
      </p>
    </div>
  );
};

export default WaveformPlayer;
