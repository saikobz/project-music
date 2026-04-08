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
};

// ตัวเล่นหลายแทร็กที่คุม WaveSurfer 4 ตัวให้เล่นและ seek ไปพร้อมกัน
export default function AdvancedMultiTrackPlayer({ baseUrl }: Props) {
  // เก็บ instance ของ WaveSurfer แยกตาม stem เพื่อให้สั่ง play, pause, seek และ setVolume ได้ทีหลัง
  const waveSurferRefs = useRef<Record<StemType, WaveSurfer | null>>({
    vocals: null,
    drums: null,
    bass: null,
    other: null,
  });
  // จำว่า pointer กำลังลาก waveform ของ stem ไหนอยู่ เพื่อทำ drag seek ให้ต่อเนื่อง
  const draggingStemRef = useRef<StemType | null>(null);

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

  useEffect(() => {
    // สร้าง waveform player แยกสำหรับแต่ละ stem ทุกครั้งที่ base URL เปลี่ยน
    stems.forEach((stem) => {
      // หา container ของ waveform ตาม id ที่ผูกกับชื่อ stem
      const container = document.getElementById(`waveform-${stem}`);
      // ถ้ายังไม่มี DOM ของ stem นี้ ก็ข้ามไปก่อน
      if (!container) return;

      // ล้างคลื่นเสียงเก่าที่ render ค้างอยู่ใน div เดิม
      container.innerHTML = "";
      // ทำลาย instance เดิมก่อน เพื่อป้องกัน player ซ้อนกันและ memory leak
      waveSurferRefs.current[stem]?.destroy();

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

      // โหลดไฟล์เสียงของ stem นี้จาก baseUrl ที่ส่งเข้ามา
      ws.load(`${baseUrl}/${stem}.wav`);
      ws.on("ready", () => {
        // เมื่อไฟล์พร้อมใช้งานแล้วค่อยเก็บ instance ลง ref
        waveSurferRefs.current[stem] = ws;
        // ตั้งค่าเสียงเริ่มต้นให้ตรงกับสถานะ mute/volume ปัจจุบัน
        ws.setVolume(mutedTracks[stem] ? 0 : trackVolumes[stem] / 100);
        // บันทึกความยาวเพลงของ stem นี้ไว้ใช้แสดงบน UI
        setDurations((prev) => ({
          ...prev,
          [stem]: ws.getDuration(),
        }));
      });
      ws.on("audioprocess", (time: number) => {
        // ระหว่างเล่นเพลงจะยิง event นี้บ่อย ๆ เพื่ออัปเดตเวลาปัจจุบันแบบ realtime
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
        waveSurferRefs.current[stem]?.destroy();
      });
    };
  }, [baseUrl]);

  useEffect(() => {
    // ทำให้สถานะ mute และ volume slider สะท้อนลงไปยัง WaveSurfer ของแต่ละแทร็กจริง
    stems.forEach((stem) => {
      // ดึง player ของ stem นี้ออกมาเพื่ออัปเดตระดับเสียงจริง
      const ws = waveSurferRefs.current[stem];
      if (!ws) return;
      // ถ้า mute ให้เสียงเป็น 0 ไม่งั้นแปลงค่าจาก 0-100 เป็น 0-1
      ws.setVolume(mutedTracks[stem] ? 0 : trackVolumes[stem] / 100);
    });
  }, [mutedTracks, trackVolumes]);

  // ตัวคำนวณกลางสำหรับแปลงการลากบน waveform ให้เป็นตำแหน่ง seek
  const seekToPointer = (stem: StemType, clientX: number) => {
    // หา player และ element ของ stem ที่กำลังถูกคลิกหรือลาก
    const ws = waveSurferRefs.current[stem];
    const container = document.getElementById(`waveform-${stem}`);
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

  const togglePlay = () => {
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
    const nextMuted = !mutedTracks[stem];
    const ws = waveSurferRefs.current[stem];

    // อัปเดต state เพื่อให้ปุ่มและ label เปลี่ยนตาม
    setMutedTracks((prev) => ({ ...prev, [stem]: nextMuted }));
    if (ws) {
      // ถ้า mute ให้เสียงเป็น 0 ไม่งั้นใช้ค่าจาก slider เดิม
      ws.setVolume(nextMuted ? 0 : trackVolumes[stem] / 100);
    }
  };

  const handleVolumeChange = (stem: StemType, value: number) => {
    // กันค่าที่หลุดช่วงจาก input โดยบังคับให้อยู่ระหว่าง 0 ถึง 100
    const nextVolume = Math.min(Math.max(value, 0), 100);
    const ws = waveSurferRefs.current[stem];

    // เก็บค่าระดับเสียงใหม่ของ stem นี้
    setTrackVolumes((prev) => ({ ...prev, [stem]: nextVolume }));
    // ถ้าปรับจนเหลือ 0 ให้ถือว่า mute อัตโนมัติ
    setMutedTracks((prev) => ({ ...prev, [stem]: nextVolume === 0 }));
    if (ws) {
      // ส่งค่าระดับเสียงใหม่ไปให้ WaveSurfer ทันที
      ws.setVolume(nextVolume === 0 ? 0 : nextVolume / 100);
    }
  };

  const handleDownload = async (stem: StemType) => {
    try {
      // ดาวน์โหลด stem จาก backend แล้วสั่ง browser ให้โหลดไฟล์ตามปกติ
      const response = await fetch(`${baseUrl}/${stem}.wav`);
      // แปลง response เป็น blob ก่อนสร้างไฟล์ดาวน์โหลด
      const blob = await response.blob();
      // สร้าง URL ชั่วคราวที่อ้างถึงไฟล์ในหน่วยความจำ
      const url = URL.createObjectURL(blob);
      // สร้างลิงก์ชั่วคราวแล้วสั่ง click เพื่อเริ่มดาวน์โหลด
      const link = document.createElement("a");
      link.href = url;
      link.download = `${stem}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      // คืนหน่วยความจำหลังดาวน์โหลดเสร็จ
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed", error);
    }
  };

  return (
    // กรอบหลักของ multitrack player
    <div className="space-y-4 rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur shadow-[0_20px_40px_rgba(17,24,39,0.45)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-semibold text-[#EDE9FE]">
          <span className="text-lg">เครื่องเล่นหลายสเตม</span>
          <span>Stem</span>
        </div>
        <div className="flex gap-2">
          {/* ปุ่มควบคุมการเล่น/หยุดของทุก stem พร้อมกัน */}
          <button
            onClick={togglePlay}
            className="rounded-lg bg-[#22D3EE] px-3 py-2 text-sm font-semibold text-black hover:bg-[#38E2FF] cursor-pointer"
          >
            {isPlaying ? "หยุดชั่วคราว" : "เล่นทั้งหมด"}
          </button>
          {/* ปุ่มรีเซ็ตทุก stem กลับไปตำแหน่งเริ่มต้น */}
          <button
            onClick={resetAll}
            className="rounded-lg border border-[#5B21B6]/50 bg-[#111827] px-3 py-2 text-sm font-semibold text-[#EDE9FE] hover:bg-[#1F2937] cursor-pointer"
          >
            เริ่มต้นใหม่
          </button>
        </div>
      </div>

      <div className="space-y-3">
        {/* วนสร้าง UI แยกให้แต่ละ stem */}
        {stems.map((stem) => (
          <div
            key={stem}
            className={`space-y-2 rounded-xl border p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)] bg-gradient-to-br ${STEM_THEME[stem].panel}`}
            style={{ borderColor: `${STEM_THEME[stem].accent}44` }}
          >
            <div className="flex items-center justify-between">
              <div className="flex flex-col">
                {/* แสดงชื่อ stem และเวลาเล่นปัจจุบันเทียบกับเวลารวม */}
                <span className="capitalize font-semibold text-[#F8FAFC]">{stem}</span>
                <span className="text-xs" style={{ color: STEM_THEME[stem].accent }}>
                  เวลา: {formatTime(currentTimes[stem])} / {durations[stem] ? formatTime(durations[stem]) : "-"}
                </span>
              </div>
              <div className="flex gap-2">
                {/* ปุ่มดาวน์โหลดไฟล์เสียงของ stem นี้ */}
                <button
                  onClick={() => handleDownload(stem)}
                  className="cursor-pointer rounded-lg bg-[#22D3EE] px-3 py-1 text-xs font-semibold text-black hover:bg-[#38E2FF]"
                >
                  ดาวน์โหลด
                </button>
                {/* ปุ่ม mute/unmute เฉพาะ stem นี้ */}
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

            <div
              className="rounded-lg border bg-[#08101C]/85 px-3 py-2"
              style={{ borderColor: `${STEM_THEME[stem].accent}33` }}
            >
              {/* ส่วนควบคุมระดับเสียงของ stem นี้ */}
              <div className="mb-2 flex items-center justify-between text-[11px] font-semibold uppercase tracking-wide">
                <span>Volume</span>
                <span
                  className="rounded-full border bg-[#0B1220] px-2 py-0.5 text-[#EDE9FE]"
                  style={{ borderColor: `${STEM_THEME[stem].accent}33`, color: STEM_THEME[stem].accent }}
                >
                  {mutedTracks[stem] ? "Mute" : `${trackVolumes[stem]}%`}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span
                  className="rounded-md border bg-[#0B1220] px-2 py-1 text-[10px] font-semibold"
                  style={{ borderColor: `${STEM_THEME[stem].accent}33`, color: STEM_THEME[stem].accent }}
                >
                  VOL
                </span>
                {/* slider จะเรียก handleVolumeChange ทุกครั้งที่ผู้ใช้ลากปรับค่า */}
                <input
                  type="range"
                  min="0"
                  max="100"
                  step="1"
                  value={trackVolumes[stem]}
                  onChange={(event) => handleVolumeChange(stem, Number(event.target.value))}
                  className="h-2 w-full cursor-pointer rounded-full bg-[#312E81]"
                  style={{ accentColor: STEM_THEME[stem].progress }}
                  aria-label={`ปรับระดับเสียงของ ${stem}`}
                />
              </div>
            </div>

            {/* div นี้เป็นพื้นที่ให้ WaveSurfer วาด waveform ลงไป */}
            <div
              id={`waveform-${stem}`}
              className="rounded-xl border bg-[#06101A] px-2 py-2 cursor-pointer shadow-[inset_0_1px_12px_rgba(255,255,255,0.03)]"
              style={{ borderColor: `${STEM_THEME[stem].accent}40` }}
              onPointerDown={(e) => {
                // เริ่ม drag seek และ seek ไปยังจุดที่กดทันที
                draggingStemRef.current = stem;
                seekToPointer(stem, e.clientX);
              }}
              onPointerMove={(e) => {
                // ถ้ายังลากอยู่ ให้ seek ตาม pointer แบบต่อเนื่อง
                if (draggingStemRef.current === stem) {
                  seekToPointer(stem, e.clientX);
                }
              }}
              onPointerUp={() => {
                // จบการลากเมื่อปล่อย pointer
                draggingStemRef.current = null;
              }}
              onPointerLeave={() => {
                // กัน pointer หลุดออกนอกพื้นที่แล้วค้างสถานะลากไว้
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
  // ถ้ายังไม่มีค่าเวลาหรือค่าผิดรูป ให้แสดงเป็น 0:00
  if (!seconds || Number.isNaN(seconds)) return "0:00";
  // แปลงวินาทีให้เป็นรูปแบบ นาที:วินาที เช่น 1:05
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${s}`;
};
