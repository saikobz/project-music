"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

// backend แยกเพลงออกมาเป็น 4 stem ชื่อคงที่เสมอ
const stems = ["vocals", "drums", "bass", "other"] as const;
type StemType = (typeof stems)[number];

// ระดับเสียงตั้งต้นของแต่ละแทร็ก เพื่อให้เริ่มฟังได้ทันทีโดยไม่ดังสุด
const DEFAULT_TRACK_VOLUMES: Record<StemType, number> = {
  vocals: 85,
  drums: 85,
  bass: 85,
  other: 85,
};

type Props = {
  baseUrl: string;
};

// ตัวเล่นหลายแทร็กที่คุม WaveSurfer 4 ตัวให้เล่นและ seek ไปพร้อมกัน
export default function AdvancedMultiTrackPlayer({ baseUrl }: Props) {
  const waveSurferRefs = useRef<Record<StemType, WaveSurfer | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  const draggingStemRef = useRef<StemType | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [mutedTracks, setMutedTracks] = useState<Record<StemType, boolean>>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  const [trackVolumes, setTrackVolumes] = useState<Record<StemType, number>>(DEFAULT_TRACK_VOLUMES);
  const [durations, setDurations] = useState<Record<StemType, number>>({
    vocals: 0,
    drums: 0,
    bass: 0,
    other: 0,
  });
  const [currentTimes, setCurrentTimes] = useState<Record<StemType, number>>({
    vocals: 0,
    drums: 0,
    bass: 0,
    other: 0,
  });

  useEffect(() => {
    // สร้าง waveform player แยกสำหรับแต่ละ stem ทุกครั้งที่ base URL เปลี่ยน
    stems.forEach((stem) => {
      const container = document.getElementById(`waveform-${stem}`);
      if (!container) return;

      container.innerHTML = "";
      waveSurferRefs.current[stem]?.destroy();

      const ws = WaveSurfer.create({
        container,
        waveColor: "#C7D2FE",
        progressColor: "#22D3EE",
        cursorColor: "#5B21B6",
        height: 70,
        barGap: 2,
        barWidth: 2,
      });

      ws.load(`${baseUrl}/${stem}.wav`);
      ws.on("ready", () => {
        waveSurferRefs.current[stem] = ws;
        ws.setVolume(mutedTracks[stem] ? 0 : trackVolumes[stem] / 100);
        setDurations((prev) => ({
          ...prev,
          [stem]: ws.getDuration(),
        }));
      });
      ws.on("audioprocess", (time: number) => {
        setCurrentTimes((prev) => ({
          ...prev,
          [stem]: time,
        }));
      });
      ws.on("seeking", (currentTime: number) => {
        setCurrentTimes((prev) => ({
          ...prev,
          [stem]: currentTime,
        }));
      });
      ws.on("finish", () => setIsPlaying(false));
    });

    return () => {
      stems.forEach((stem) => {
        waveSurferRefs.current[stem]?.destroy();
      });
    };
  }, [baseUrl]);

  useEffect(() => {
    // ทำให้สถานะ mute และ volume slider สะท้อนลงไปยัง WaveSurfer ของแต่ละแทร็กจริง
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (!ws) return;
      ws.setVolume(mutedTracks[stem] ? 0 : trackVolumes[stem] / 100);
    });
  }, [mutedTracks, trackVolumes]);

  // ตัวคำนวณกลางสำหรับแปลงการลากบน waveform ให้เป็นตำแหน่ง seek
  const seekToPointer = (stem: StemType, clientX: number) => {
    const ws = waveSurferRefs.current[stem];
    const container = document.getElementById(`waveform-${stem}`);
    if (!ws || !container) return;
    const rect = container.getBoundingClientRect();
    const progress = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    ws.seekTo(progress);
    setCurrentTimes((prev) => ({
      ...prev,
      [stem]: progress * ws.getDuration(),
    }));
  };

  const togglePlay = () => {
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (ws) ws.playPause();
    });
    setIsPlaying((prev) => !prev);
  };

  const resetAll = () => {
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (ws) {
        ws.pause();
        ws.seekTo(0);
      }
    });
    setIsPlaying(false);
    setCurrentTimes({
      vocals: 0,
      drums: 0,
      bass: 0,
      other: 0,
    });
  };

  const toggleMute = (stem: StemType) => {
    const nextMuted = !mutedTracks[stem];
    const ws = waveSurferRefs.current[stem];

    setMutedTracks((prev) => ({ ...prev, [stem]: nextMuted }));
    if (ws) {
      ws.setVolume(nextMuted ? 0 : trackVolumes[stem] / 100);
    }
  };

  const handleVolumeChange = (stem: StemType, value: number) => {
    const nextVolume = Math.min(Math.max(value, 0), 100);
    const ws = waveSurferRefs.current[stem];

    setTrackVolumes((prev) => ({ ...prev, [stem]: nextVolume }));
    setMutedTracks((prev) => ({ ...prev, [stem]: nextVolume === 0 }));
    if (ws) {
      ws.setVolume(nextVolume === 0 ? 0 : nextVolume / 100);
    }
  };

  const handleDownload = async (stem: StemType) => {
    try {
      // ดาวน์โหลด stem จาก backend แล้วสั่ง browser ให้โหลดไฟล์ตามปกติ
      const response = await fetch(`${baseUrl}/${stem}.wav`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${stem}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed", error);
    }
  };

  return (
    <div className="space-y-4 rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur shadow-[0_20px_40px_rgba(17,24,39,0.45)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-semibold text-[#EDE9FE]">
          <span className="text-lg">เครื่องเล่นหลายสเตม</span>
          <span>Stem</span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={togglePlay}
            className="rounded-lg bg-[#22D3EE] px-3 py-2 text-sm font-semibold text-black hover:bg-[#38E2FF] cursor-pointer"
          >
            {isPlaying ? "หยุดชั่วคราว" : "เล่นทั้งหมด"}
          </button>
          <button
            onClick={resetAll}
            className="rounded-lg border border-[#5B21B6]/50 bg-[#111827] px-3 py-2 text-sm font-semibold text-[#EDE9FE] hover:bg-[#1F2937] cursor-pointer"
          >
            เริ่มต้นใหม่
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {stems.map((stem) => (
          <div key={stem} className="space-y-2 rounded-xl border border-[#5B21B6]/30 bg-[#111827] p-3">
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                <span className="capitalize font-semibold text-[#EDE9FE]">{stem}</span>
                <span className="text-xs text-[#A78BFA]">
                  เวลา: {formatTime(currentTimes[stem])} / {durations[stem] ? formatTime(durations[stem]) : "-"}
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleDownload(stem)}
                  className="cursor-pointer rounded-lg bg-[#22D3EE] px-3 py-1 text-xs font-semibold text-black hover:bg-[#38E2FF]"
                >
                  ดาวน์โหลด
                </button>
                <button
                  onClick={() => toggleMute(stem)}
                  className={`cursor-pointer rounded-lg border px-3 py-1 text-xs font-semibold ${
                    mutedTracks[stem]
                      ? "bg-[#246e41] border-[#5B21B6]/60 text-[#EDE9FE]"
                      : "bg-[#b62121] border-[#22D3EE]/40 text-white"
                  }`}
                >
                  {mutedTracks[stem] ? "เปิดเสียง" : "ปิดเสียง"}
                </button>
              </div>
            </div>

            <div className="rounded-lg border border-[#5B21B6]/25 bg-[#0B1021]/80 px-3 py-2">
              <div className="mb-2 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide text-[#A78BFA]">
                <span>Volume</span>
                <span className="rounded-full border border-[#22D3EE]/30 bg-[#111827] px-2 py-0.5 text-[#EDE9FE]">
                  {mutedTracks[stem] ? "Mute" : `${trackVolumes[stem]}%`}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="rounded-md border border-[#22D3EE]/30 bg-[#111827] px-2 py-1 text-[10px] font-semibold text-[#22D3EE]">
                  VOL
                </span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={trackVolumes[stem]}
                  onChange={(event) => handleVolumeChange(stem, Number(event.target.value))}
                  className="h-2 w-full cursor-pointer rounded-full bg-[#312E81]"
                  style={{ accentColor: "#22D3EE" }}
                  aria-label={`ปรับระดับเสียงของ ${stem}`}
                />
              </div>
            </div>

            <div
              id={`waveform-${stem}`}
              className="rounded-lg bg-[#0B1021] cursor-pointer"
              onPointerDown={(e) => {
                draggingStemRef.current = stem;
                seekToPointer(stem, e.clientX);
              }}
              onPointerMove={(e) => {
                if (draggingStemRef.current === stem) {
                  seekToPointer(stem, e.clientX);
                }
              }}
              onPointerUp={() => {
                draggingStemRef.current = null;
              }}
              onPointerLeave={() => {
                draggingStemRef.current = null;
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

// แปลงวินาทีให้อยู่ในรูป m:ss สำหรับแสดงเวลาใต้แต่ละ stem
const formatTime = (seconds: number) => {
  if (!seconds || Number.isNaN(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};
