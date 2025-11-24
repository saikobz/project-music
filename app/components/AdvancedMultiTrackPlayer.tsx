"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

const stems = ["vocals", "drums", "bass", "other"] as const;
type StemType = (typeof stems)[number];

type Props = {
  baseUrl: string;
};

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
        responsive: true,
      });

      ws.load(`${baseUrl}/${stem}.wav`);
      ws.on("ready", () => {
        waveSurferRefs.current[stem] = ws;
        if (mutedTracks[stem]) ws.setVolume(0);
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
      ws.on("seek", (progress: number) => {
        const dur = ws.getDuration();
        setCurrentTimes((prev) => ({
          ...prev,
          [stem]: progress * dur,
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
    const ws = waveSurferRefs.current[stem];
    const isMuted = !mutedTracks[stem];
    if (ws) {
      ws.setVolume(isMuted ? 0 : 1);
      setMutedTracks((prev) => ({ ...prev, [stem]: isMuted }));
    }
  };

  const handleDownload = async (stem: StemType) => {
    try {
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
        <div className="flex items-center gap-2 text-[#EDE9FE] font-semibold">
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
            className="rounded-lg bg-[#111827] border border-[#5B21B6]/50 px-3 py-2 text-sm font-semibold text-[#EDE9FE] hover:bg-[#1F2937] cursor-pointer"
          >
            เริ่มต้นใหม่
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {stems.map((stem) => (
          <div key={stem} className="space-y-2 rounded-xl border border-[#5B21B6]/30 bg-[#111827] p-3">
            <div className="flex justify-between items-center">
              <div className="flex flex-col">
                <span className="capitalize font-semibold text-[#EDE9FE]">{stem}</span>
                <span className="text-xs text-[#A78BFA]">
                  เวลา: {formatTime(currentTimes[stem])} / {durations[stem] ? formatTime(durations[stem]) : "-"}
                </span>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleDownload(stem)}
                  className="text-xs px-3 py-1 rounded-lg bg-[#22D3EE] text-black font-semibold hover:bg-[#38E2FF] cursor-pointer"
                >
                  ดาวน์โหลด
                </button>
                <button
                  onClick={() => toggleMute(stem)}
                  className={`text-xs px-3 py-1 rounded-lg border font-semibold cursor-pointer ${
                    mutedTracks[stem]
                      ? "bg-[#0B1021] border-[#5B21B6]/60 text-[#EDE9FE]"
                      : "bg-[#5B21B6] border-[#22D3EE]/40 text-white"
                  }`}
                >
                  {mutedTracks[stem] ? "Unmute" : "Mute"}
                </button>
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

const formatTime = (seconds: number) => {
  if (!seconds || Number.isNaN(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};
