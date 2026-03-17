"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

interface WaveformPlayerProps {
  audioUrl: string;
}

// ตัวเล่น waveform สำหรับไฟล์ WAV เดี่ยวที่ผ่านการประมวลผลแล้ว
const WaveformPlayer: React.FC<WaveformPlayerProps> = ({ audioUrl }) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const waveSurferRef = useRef<WaveSurfer | null>(null);
  const isDraggingRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(85);

  useEffect(() => {
    if (!containerRef.current) return;

    // สร้าง WaveSurfer ใหม่ทุกครั้งเมื่อมีไฟล์ผลลัพธ์ตัวใหม่ถูกโหลดเข้ามา
    waveSurferRef.current?.destroy();

    waveSurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#C7D2FE",
      progressColor: "#5B21B6",
      cursorColor: "#22D3EE",
      height: 140,
      barGap: 2,
      barWidth: 2,
    });

    waveSurferRef.current.load(audioUrl);
    waveSurferRef.current.setVolume(volume / 100);
    waveSurferRef.current.on("finish", () => setIsPlaying(false));
    waveSurferRef.current.on("error", (e) => {
      // ปล่อยผ่าน error ที่เกิดจากการ destroy หรือ abort ระหว่างโหลดไฟล์
      if ((e as { name?: string })?.name === "AbortError") return;
      console.error("WaveSurfer error", e);
    });

    return () => {
      waveSurferRef.current?.destroy();
    };
  }, [audioUrl]);

  useEffect(() => {
    waveSurferRef.current?.setVolume(volume / 100);
  }, [volume]);

  // แปลงตำแหน่ง pointer บน waveform ให้เป็นตำแหน่ง seek ของเสียง
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

  const handleVolumeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const nextVolume = Number(event.target.value);
    setVolume(nextVolume);
    waveSurferRef.current?.setVolume(nextVolume / 100);
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
          รีเซ็ต
        </button>
      </div>
      <div className="rounded-xl border border-[#7C3AED]/25 bg-[#120E22] px-3 py-2">
        <div className="mb-2 flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-[#A78BFA]">
          <span>Volume</span>
          <span className="rounded-full border border-[#22D3EE]/35 bg-[#0F172A] px-2 py-0.5 text-[#EDE9FE]">{volume}%</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-md border border-[#22D3EE]/35 bg-[#0F172A] px-2 py-1 text-[10px] font-semibold text-[#22D3EE]">
            VOL
          </span>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={volume}
            onChange={handleVolumeChange}
            className="h-2 w-full cursor-pointer rounded-full bg-[#312E81]"
            style={{ accentColor: "#22D3EE" }}
            aria-label="ปรับระดับเสียงไฟล์ที่ประมวลผลแล้ว"
          />
        </div>
      </div>
      <p className="text-center text-xs text-[#EDE9FE]/70">เลื่อนเพื่อ seek ไปยังตำแหน่งที่ต้องการบน waveform ได้ทันที</p>
    </div>
  );
};

export default WaveformPlayer;
