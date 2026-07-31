"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { API_BASE_URL, DEFAULT_STEMS } from "@/lib/config";

// backend แยกเพลงออกมาเป็น 4 stem ชื่อคงที่เสมอ
const stems = DEFAULT_STEMS;
type StemType = (typeof stems)[number];

// ระดับเสียงตั้งต้นของแต่ละแทร็ก เพื่อให้เริ่มฟังได้ทันทีโดยไม่ดังสุด
const DEFAULT_TRACK_VOLUMES: Record<StemType, number> = {
  vocals: 85,
  drums: 85,
  bass: 85,
  other: 85,
};

const STEM_THEME: Record<StemType, { wave: string; progress: string; accent: string; panel: string }> = {
  vocals: {
    wave: "#FBCFE8",
    progress: "#F472B6",
    accent: "#F9A8D4",
    panel: "from-[#2A1321] to-[#140C19]",
  },
  drums: {
    wave: "#FDE68A",
    progress: "#F59E0B",
    accent: "#FBBF24",
    panel: "from-[#2B1E0D] to-[#161008]",
  },
  bass: {
    wave: "#A7F3D0",
    progress: "#10B981",
    accent: "#34D399",
    panel: "from-[#11261F] to-[#0A1512]",
  },
  other: {
    wave: "#BFDBFE",
    progress: "#38BDF8",
    accent: "#7DD3FC",
    panel: "from-[#122234] to-[#0A111C]",
  },
};

type Props = {
  baseUrl: string;
  fileId?: string; // เพิ่ม fileId สำหรับเรียก API ประมวลผล
};

// ตัวเล่นหลายแทร็กที่คุม WaveSurfer 4 ตัวให้เล่นและ seek ไปพร้อมกัน
export default function AdvancedMultiTrackPlayer({ baseUrl, fileId }: Props) {
  // เก็บ instance ของ WaveSurfer แยกตาม stem เพื่อให้สั่ง play, pause, seek และ setVolume ได้ทีหลัง
  const waveSurferRefs = useRef<Record<StemType, WaveSurfer | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  // จำว่า pointer กำลังลาก waveform ของ stem ไหนอยู่ เพื่อทำ drag seek ให้ต่อเนื่อง
  const draggingStemRef = useRef<StemType | null>(null);
  // เก็บเวลาครั้งสุดท้ายที่อัปเดตตัวเลขเวลา (throttle audioprocess — M5)
  const lastTimeUpdateRef = useRef(0);

  // บอกว่า player โดยรวมอยู่ในสถานะเล่นอยู่หรือหยุดอยู่
  const [isPlaying, setIsPlaying] = useState(false);
  // เก็บสถานะ mute ของแต่ละ stem แยกกัน
  const [mutedTracks, setMutedTracks] = useState<Record<StemType, boolean>>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  // เก็บระดับเสียงของแต่ละ stem เป็นเปอร์เซ็นต์ 0-100
  const [trackVolumes, setTrackVolumes] = useState<Record<StemType, number>>(DEFAULT_TRACK_VOLUMES);
  // เก็บความยาวรวมของไฟล์แต่ละ stem ไว้แสดงผลบนหน้าจอ
  const [durations, setDurations] = useState<Record<StemType, number>>({
    vocals: 0,
    drums: 0,
    bass: 0,
    other: 0,
  });
  // เก็บเวลาปัจจุบันของแต่ละ stem เพื่ออัปเดตตัวเลขระหว่างเล่นหรือ seek
  const [currentTimes, setCurrentTimes] = useState<Record<StemType, number>>({
    vocals: 0,
    drums: 0,
    bass: 0,
    other: 0,
  });

  // สถานะสำหรับ Vocal Polish
  const [isVocalPolished, setIsVocalPolished] = useState(false);
  const [isPolishing, setIsPolishing] = useState(false);

  // สถานะสำหรับ Solo (เก็บได้ทีละ 1 แทร็ก)
  const [soloedTrack, setSoloedTrack] = useState<StemType | null>(null);

  // ref สำหรับ container ของ WaveSurfer แต่ละ stem (หลีกเลี่ยงการแตะ DOM ที่ React จัดการ)
  const containerRefs = useRef<Record<StemType, HTMLDivElement | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  // เก็บ state ล่าสุดไว้ใน ref เพื่อให้ handler ของ WaveSurfer อ่านค่าได้ไม่ stale
  const latestStateRef = useRef({ mutedTracks, trackVolumes, soloedTrack });
  latestStateRef.current = { mutedTracks, trackVolumes, soloedTrack };
  // ตรวจว่าแต่ละ stem โหลดพร้อมเล่นแล้วหรือยัง (กันกด Play ก่อนพร้อม -> stems หลุด sync)
  const [readyMap, setReadyMap] = useState<Record<StemType, boolean>>({
    vocals: false,
    drums: false,
    bass: false,
    other: false,
  });
  // ข้อความ error ของ Vocal Polish เพื่อให้ผู้ใช้เห็นว่าเหตุผลอะไรที่ปรับไม่ได้
  const [polishError, setPolishError] = useState<string | null>(null);
  const polishAbortRef = useRef<AbortController | null>(null);
  // abort fetch ค้างไว้ตอน component ถูกถอด เพื่อไม่ให้ setState หลัง unmount
  useEffect(() => () => polishAbortRef.current?.abort(), []);

  // ตั้งค่า volume ของ ws ให้ตรงกับสถานะ solo/mute/volume ล่าสุด (ใช้ ref เพื่อกัน stale closure)
  const syncVolume = (ws: WaveSurfer, stem: StemType) => {
    const { mutedTracks: muted, trackVolumes: volumes, soloedTrack: soloed } = latestStateRef.current;
    const shouldPlay = soloed !== null ? soloed === stem : !muted[stem];
    ws.setVolume(shouldPlay ? volumes[stem] / 100 : 0);
  };

  useEffect(() => {
    // สร้าง waveform player แยกสำหรับแต่ละ stem ทุกครั้งที่ base URL เปลี่ยน
    // (รีเซ็ตสถานะพร้อมเล่นใหม่ทุกครั้งที่สร้างชุด player ใหม่)
    setReadyMap({ vocals: false, drums: false, bass: false, other: false });

    stems.forEach((stem) => {
      const container = containerRefs.current[stem];
      if (!container) return;

      // ทำลาย instance เดิมก่อน เพื่อป้องกัน player ซ้อนกันและ memory leak
      const previous = waveSurferRefs.current[stem];
      if (previous) {
        previous.destroy();
        waveSurferRefs.current[stem] = null;
      }

      // สร้าง WaveSurfer ตัวใหม่ของ stem นี้ พร้อมกำหนดหน้าตา waveform
      const ws = WaveSurfer.create({
        container,
        waveColor: STEM_THEME[stem].wave,
        progressColor: STEM_THEME[stem].progress,
        cursorColor: STEM_THEME[stem].accent,
        height: 72,
        barGap: 1.75,
        barWidth: 2,
        barRadius: 999,
        normalize: true,
      });

      // โหลดไฟล์เสียงของ stem นี้จาก baseUrl ที่ส่งเข้ามา (ถ้าเป็น vocals และถูก polish ให้ใช้ไฟล์ polished)
      const audioUrl = (stem === "vocals" && isVocalPolished)
        ? `${baseUrl}/vocals_polished.wav`
        : `${baseUrl}/${stem}.wav`;
      ws.load(audioUrl).catch((e: unknown) => {
        if ((e as { name?: string })?.name === "AbortError") return;
        console.error("WaveSurfer load error for", stem, e);
      });
      ws.on("ready", () => {
        // เมื่อไฟล์พร้อมใช้งานแล้วค่อยเก็บ instance ลง ref
        waveSurferRefs.current[stem] = ws;
        // ตั้งค่าเสียงเริ่มต้นให้ตรงกับสถานะ solo/mute/volume ปัจจุบัน
        syncVolume(ws, stem);
        setReadyMap((prev) => ({ ...prev, [stem]: true }));
        // บันทึกความยาวเพลงของ stem นี้ไว้ใช้แสดงบน UI
        setDurations((prev) => ({
          ...prev,
          [stem]: ws.getDuration(),
        }));
      });
      ws.on("audioprocess", (time: number) => {
        // ระหว่างเล่นเพลงจะยิง event นี้ทุก animation frame (~60fps)
        // throttle เหลือ ~10 ครั้ง/วินาที เพื่อไม่ให้ setState 240 ครั้ง/วินาที
        // (ตัวเลขเวลาที่แสดงไม่จำเป็นต้องละเอียดขนาดนั้น — M5)
        const now = performance.now();
        if (now - lastTimeUpdateRef.current < 100) return;
        lastTimeUpdateRef.current = now;
        setCurrentTimes((prev) => ({
          ...prev,
          [stem]: time,
        }));
      });
      ws.on("seeking", (currentTime: number) => {
        // ตอนผู้ใช้เลื่อนตำแหน่งเล่น ให้เวลาใน UI เปลี่ยนตามทันที
        setCurrentTimes((prev) => ({
          ...prev,
          [stem]: currentTime,
        }));
      });
      // เมื่อเล่นจนจบ ให้ปุ่มเล่นกลับไปเป็นสถานะหยุด
      ws.on("finish", () => setIsPlaying(false));
    });

    return () => {
      // cleanup ตอน component ถูกถอด หรือก่อน effect ทำงานรอบใหม่
      stems.forEach((stem) => {
        const ws = waveSurferRefs.current[stem];
        if (ws) {
          ws.destroy();
          waveSurferRefs.current[stem] = null;
        }
      });
    };
  }, [baseUrl, fileId]);

  // เมื่อเปิด/ปิด Vocal Polish ให้โหลดเฉพาะ stem vocals ใหม่
  // (ไม่ต้อง rebuild player ทั้ง 4 ตัว เพื่อไม่ให้เสียตำแหน่งเล่นของ stem อื่น)
  useEffect(() => {
    const ws = waveSurferRefs.current.vocals;
    if (!ws) return;
    const audioUrl = `${baseUrl}/vocals${isVocalPolished ? "_polished" : ""}.wav`;
    ws.load(audioUrl).catch((e: unknown) => {
      if ((e as { name?: string })?.name === "AbortError") return;
      console.error("WaveSurfer reload error for vocals", e);
    });
  }, [isVocalPolished, baseUrl]);

  useEffect(() => {
    // ทำให้สถานะ mute, solo และ volume slider สะท้อนลงไปยัง WaveSurfer ของแต่ละแทร็กจริง
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (ws) syncVolume(ws, stem);
    });
  }, [mutedTracks, trackVolumes, soloedTrack]);

  // ตัวคำนวณกลางสำหรับแปลงการลากบน waveform ให้เป็นตำแหน่ง seek
  const seekToPointer = (stem: StemType, clientX: number) => {
    // หา player และ element ของ stem ที่กำลังถูกคลิกหรือลาก
    const ws = waveSurferRefs.current[stem];
    const container = containerRefs.current[stem];
    if (!ws || !container) return;
    // อ่านขนาดและตำแหน่งจริงของ waveform บนหน้าจอ
    const rect = container.getBoundingClientRect();
    // แปลงตำแหน่ง pointer จาก pixel ให้เป็นสัดส่วน 0-1 สำหรับสั่ง seek
    const progress = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    // เลื่อนไปยังตำแหน่งใหม่ในไฟล์เสียง
    ws.seekTo(progress);
    // อัปเดตเวลาที่แสดงบนหน้าจอให้สัมพันธ์กับตำแหน่งใหม่
    setCurrentTimes((prev) => ({
      ...prev,
      [stem]: progress * ws.getDuration(),
    }));
  };

  // ปุ่มเล่นจะใช้ได้ก็ต่อเมื่อทุก stem โหลดพร้อมแล้ว (กันกดเร็วเกิน -> หลุด sync ถาวร)
  const allReady = stems.every((stem) => readyMap[stem]);

  const togglePlay = () => {
    if (!allReady) return;
    // สลับ play/pause ของทุก stem พร้อมกัน เพื่อให้ยัง sync กันอยู่
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (ws) ws.playPause();
    });
    // สลับสถานะปุ่มใน UI
    setIsPlaying((prev) => !prev);
  };

  const resetAll = () => {
    // หยุดและเลื่อนทุก stem กลับไปต้นเพลง
    stems.forEach((stem) => {
      const ws = waveSurferRefs.current[stem];
      if (ws) {
        ws.pause();
        ws.seekTo(0);
      }
    });
    // รีเซ็ต state ให้ตัวเลขเวลาบนหน้าจอกลับเป็นศูนย์ทั้งหมด
    setIsPlaying(false);
    setCurrentTimes({
      vocals: 0,
      drums: 0,
      bass: 0,
      other: 0,
    });
  };

  const toggleMute = (stem: StemType) => {
    // สลับค่า mute ของ stem นี้จากค่าเดิม
    setMutedTracks((prev) => ({ ...prev, [stem]: !prev[stem] }));
  };

  const toggleSolo = (stem: StemType) => {
    // สลับค่า solo ของ stem นี้ (ถ้าคลิกซ้ำให้ปิดโซโล่ ถ้าคลิกตัวอื่นให้สลับไปโซโล่ตัวใหม่แทน)
    setSoloedTrack((prev) => (prev === stem ? null : stem));
  };

  const handleVolumeChange = (stem: StemType, value: number) => {
    // กันค่าที่หลุดช่วงจาก input โดยบังคับให้อยู่ระหว่าง 0 ถึง 100
    const nextVolume = Math.min(Math.max(value, 0), 100);

    // เก็บค่าระดับเสียงใหม่ของ stem นี้
    setTrackVolumes((prev) => ({ ...prev, [stem]: nextVolume }));
    // ถ้าปรับจนเหลือ 0 ให้ถือว่า mute อัตโนมัติ
    setMutedTracks((prev) => ({ ...prev, [stem]: nextVolume === 0 }));
  };



  const handleToggleVocalPolish = async () => {
    if (!fileId) return;

    // ถ้าเคย polish แล้ว และกดอีกครั้งให้ปิด
    if (isVocalPolished) {
      setIsVocalPolished(false);
      setPolishError(null);
      return;
    }

    // ถ้ายังไม่เคย polish ให้เรียก API
    setIsPolishing(true);
    setPolishError(null);
    polishAbortRef.current?.abort();
    const controller = new AbortController();
    polishAbortRef.current = controller;
    try {
      const apiBase = API_BASE_URL;
      const res = await fetch(`${apiBase}/api/process/vocal-polish?file_id=${fileId}`, {
        method: "POST",
        signal: controller.signal,
      });
      if (!res.ok) {
        // แสดงข้อความ error จาก backend (ถ้ามี) ให้ผู้ใช้รู้ว่าล้มเหลว
        let message = "เกิดข้อผิดพลาดในการปรับแต่งเสียงร้อง";
        try {
          const body = await res.json();
          if (typeof body?.detail === "string") message = body.detail;
        } catch {
          // อ่าน body ไม่ได้ ใช้ข้อความเริ่มต้น
        }
        setPolishError(message);
        return;
      }
      setIsVocalPolished(true);
    } catch (err) {
      if ((err as { name?: string })?.name === "AbortError") return;
      console.error("Failed to polish vocals:", err);
      setPolishError("ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ได้ กรุณาลองใหม่");
    } finally {
      setIsPolishing(false);
    }
  };



  return (
    <div className="space-y-6 rounded-2xl border border-[#2C2824] bg-[#0D0B0A] p-6 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
      <div className="flex items-center justify-between border-b border-[#2C2824] pb-5">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-[#E5A93D]/20 to-[#E5A93D]/5 border border-[#E5A93D]/20">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-[#E5A93D]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white">Stem Mixer</h2>
            <p className="text-xs font-medium uppercase tracking-widest text-[#8E8E8E] mt-0.5">Studio Grade Playback</p>
          </div>
        </div>
        <div className="flex gap-3">
          <button
            onClick={resetAll}
            className="rounded-xl border border-[#36322E] bg-[#161412] px-5 py-2.5 text-sm font-semibold text-[#8E8E8E] transition-all hover:border-[#5C5854] hover:text-white"
          >
            Reset
          </button>
          <button
            onClick={togglePlay}
            disabled={!allReady}
            data-testid="play-toggle"
            className={`flex items-center justify-center gap-2 rounded-xl px-6 py-2.5 text-sm font-bold transition-all min-w-[120px] disabled:cursor-not-allowed disabled:opacity-40 ${
              isPlaying
                ? "bg-[#2C2824] text-white hover:bg-[#36322E] shadow-inner"
                : "bg-gradient-to-b from-[#E5A93D] to-[#D6962A] text-[#0D0B0A] shadow-[0_0_20px_rgba(229,169,61,0.2)] hover:shadow-[0_0_25px_rgba(229,169,61,0.4)] hover:to-[#E5A93D]"
            }`}
          >
            {isPlaying ? "Pause" : "Play All"}
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {stems.map((stem) => {
          const isMuted = mutedTracks[stem];
          const isSoloed = soloedTrack === stem;
          const isDimmed = soloedTrack !== null && !isSoloed;

          return (
            <div
              key={stem}
              className={`relative overflow-hidden rounded-2xl border transition-all duration-300 ${
                isDimmed 
                  ? "border-[#161412] opacity-30 grayscale" 
                  : isSoloed 
                    ? "border-[#E5A93D]/50 bg-[#161412] shadow-[0_0_30px_rgba(229,169,61,0.05)]" 
                    : "border-[#2C2824] bg-[#161412] hover:border-[#36322E]"
              }`}
            >
              {/* Subtle background tint based on stem color */}
              <div 
                className="absolute inset-0 opacity-[0.03]" 
                style={{ backgroundColor: STEM_THEME[stem].accent }} 
              />
              
              <div className="relative flex flex-col md:flex-row p-4 gap-5">
                {/* Control Panel (Left) */}
                <div className="flex w-full md:w-56 flex-shrink-0 flex-col justify-between border-b border-[#2C2824] pb-4 md:border-b-0 md:border-r md:pb-0 md:pr-5">
                  <div className="flex items-center justify-between mb-4 mt-1">
                    <span className="text-[13px] font-bold uppercase tracking-widest text-white drop-shadow-md">
                      {stem}
                    </span>
                    <span className="text-[11px] font-mono font-medium text-[#8E8E8E] bg-[#000] px-2 py-0.5 rounded-md border border-[#2C2824]">
                      {formatTime(currentTimes[stem])}
                    </span>
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={() => toggleMute(stem)}
                      data-testid={`mute-${stem}`}
                      className={`flex-1 rounded-lg border py-2 text-[11px] font-bold tracking-wider transition-all ${
                        isMuted
                          ? "border-[#EF4444] bg-[#EF4444]/10 text-[#EF4444] shadow-[0_0_10px_rgba(255,68,68,0.15)]"
                          : "border-[#36322E] bg-[#0D0B0A] text-[#8E8E8E] hover:border-[#5C5854] hover:text-white"
                      }`}
                    >
                      MUTE
                    </button>
                    <button
                      onClick={() => toggleSolo(stem)}
                      data-testid={`solo-${stem}`}
                      className={`flex-1 rounded-lg border py-2 text-[11px] font-bold tracking-wider transition-all ${
                        isSoloed
                          ? "border-[#E5A93D] bg-[#E5A93D] text-[#0D0B0A] shadow-[0_0_15px_rgba(229,169,61,0.3)]"
                          : "border-[#36322E] bg-[#0D0B0A] text-[#8E8E8E] hover:border-[#E5A93D]/50 hover:text-[#E5A93D]"
                      }`}
                    >
                      SOLO
                    </button>
                    {stem === "vocals" && (
                      <button
                        onClick={handleToggleVocalPolish}
                        disabled={isPolishing || !readyMap.vocals}
                        data-testid="polish-toggle"
                        title="AI Vocal Polish"
                        className={`flex-1 flex items-center justify-center rounded-lg border py-2 transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
                          isVocalPolished
                            ? "border-purple-500 bg-purple-500/15 text-purple-400 shadow-[0_0_15px_rgba(168,85,247,0.15)]"
                            : "border-[#36322E] bg-[#0D0B0A] text-[#8E8E8E] hover:border-purple-500 hover:text-purple-400"
                        }`}
                      >
                        {isPolishing ? (
                          <span className="h-3 w-3 animate-spin rounded-full border-2 border-purple-500 border-t-transparent"></span>
                        ) : (
                          <span className="text-sm leading-none">✨</span>
                        )}
                      </button>
                    )}
                  </div>
                  {stem === "vocals" && polishError && (
                    <p data-testid="polish-error" className="mt-2 text-[11px] font-medium text-[#EF4444]">
                      {polishError}
                    </p>
                  )}
                </div>

                {/* Waveform & Volume (Right) */}
                <div className="flex flex-1 flex-col justify-center gap-3">
                  <div className="flex items-center gap-4 rounded-xl border border-[#2C2824] bg-[#050505] p-2 px-4">
                    <div className="flex items-center gap-1.5 w-14">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5 text-[#5C5854]" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" /></svg>
                      <span className="text-[10px] font-bold text-white font-mono">{String(trackVolumes[stem]).padStart(3, "0")}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="1"
                      value={trackVolumes[stem]}
                      onChange={(event) => handleVolumeChange(stem, Number(event.target.value))}
                      className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-[#36322E]"
                      style={{ accentColor: STEM_THEME[stem].progress }}
                      aria-label={`Volume for ${stem}`}
                    />
                  </div>
                  
                  <div
                    className="relative h-[72px] w-full cursor-pointer overflow-hidden rounded-xl border border-[#2C2824] bg-[#050505] shadow-inner"
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
                  >
                    {/* container ของ WaveSurfer แยกจาก DOM ที่ React render (ห้าม innerHTML ล้าง) */}
                    <div
                      ref={(el) => {
                        containerRefs.current[stem] = el;
                      }}
                      className="h-full w-full"
                    />
                    {/* Gradient overlay เป็น sibling ของ WaveSurfer container เพื่อให้ React ดูแลได้ */}
                    <div className="pointer-events-none absolute inset-0 z-10 bg-gradient-to-b from-black/40 via-transparent to-black/40" />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// แปลงวินาทีให้อยู่ในรูป m:ss สำหรับแสดงเวลาใต้แต่ละ stem
const formatTime = (seconds: number) => {
  // ถ้ายังไม่มีค่าเวลาหรือค่าผิดรูป ให้แสดงเป็น 0:00
  if (!seconds || Number.isNaN(seconds)) return "0:00";
  // แปลงวินาทีให้เป็นรูปแบบ นาที:วินาที เช่น 1:05
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};
