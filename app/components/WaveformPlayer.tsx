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
      waveColor: "#333333",
      progressColor: "#E5A93D",
      cursorColor: "#FFFFFF",
      height: 96,
      barGap: 2,
      barWidth: 2,
      barRadius: 999,
      normalize: true,
    });

    // .load() คืน Promise ที่จะ reject เป็น AbortError เมื่อ effect cleanup
    // (เช่น audioUrl เปลี่ยน หรือ React Strict Mode remount) — กลบทิ้งไว้
    waveSurferRef.current.load(audioUrl).catch((e: unknown) => {
      if ((e as { name?: string })?.name === "AbortError") return;
      console.error("WaveSurfer load error", e);
    });
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
    <div className="space-y-4 rounded-2xl border border-[#222] bg-[#0A0A0A] p-5 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
      <div className="flex items-center justify-between border-b border-[#222] pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#E5A93D]/10 border border-[#E5A93D]/20">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#E5A93D]" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" />
            </svg>
          </div>
          <div>
            <span className="text-sm font-bold uppercase tracking-widest text-white">Processed Audio</span>
            <div className="text-[10px] text-[#8E8E8E] font-medium mt-0.5">READY TO PLAY</div>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={stopPlayback}
            className="rounded-xl border border-[#333] bg-[#121212] px-4 py-2 text-xs font-semibold text-[#8E8E8E] transition-all hover:border-[#555] hover:text-white"
          >
            Reset
          </button>
          <button
            onClick={togglePlay}
            className={`flex min-w-[90px] items-center justify-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition-all ${
              isPlaying
                ? "bg-[#222] text-white hover:bg-[#333] shadow-inner"
                : "bg-gradient-to-br from-[#E5A93D] to-[#D6962A] text-[#0A0A0A] shadow-[0_0_15px_rgba(229,169,61,0.2)] hover:shadow-[0_0_20px_rgba(229,169,61,0.3)] hover:from-[#F3C05D]"
            }`}
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {/* Waveform container */}
        <div
          ref={containerRef}
          className="relative h-[96px] w-full cursor-pointer overflow-hidden rounded-xl border border-[#222] bg-[#050505] shadow-inner"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-transparent to-black/40 pointer-events-none z-10" />
        </div>

        {/* Volume control */}
        <div className="flex items-center gap-4 rounded-xl border border-[#222] bg-[#111] px-4 py-3">
          <div className="flex w-16 items-center gap-2 border-r border-[#333] pr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-[#555]" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" /></svg>
            <span className="text-xs font-bold text-white font-mono">{String(volume).padStart(3, "0")}</span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={volume}
            onChange={handleVolumeChange}
            className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-[#333]"
            style={{ 
              background: `linear-gradient(to right, #E5A93D ${volume}%, #333 ${volume}%)`
            }}
            aria-label="Volume"
          />
          <style>{`
            input[type=range]::-webkit-slider-thumb {
              -webkit-appearance: none;
              height: 12px;
              width: 12px;
              border-radius: 50%;
              background: #fff;
              box-shadow: 0 0 10px rgba(229,169,61,0.5);
              margin-top: -5.25px;
            }
          `}</style>
        </div>
      </div>
    </div>
  );
};

export default WaveformPlayer;
