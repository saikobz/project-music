"use client";

import Image from "next/image";
import React, { useEffect, useMemo, useRef, useState } from "react";

type GenreId = "pop" | "rock" | "trap" | "country" | "soul";

type EqBand = {
  label: string;
  freq: number;
  gain: number;
  q: number;
};

type GenreEqPreset = {
  id: GenreId;
  title: string;
  subtitle: string;
  accent: string;
  profileLabel: string;
  graphImage: string;
  bands: EqBand[];
};

// ข้อมูล preset แบบคงที่ ใช้สำหรับอธิบายแนวคิดการปรับ EQ ตาม genre
const GENRE_PRESETS: GenreEqPreset[] = [
  {
    id: "pop",
    title: "Pop",
    subtitle: "โฟกัส vocal ให้เด่น และเติม high ให้ airy.",
    accent: "#22D3EE",
    profileLabel: "Vocal Presence",
    graphImage: "/eq-graphs/pop.png",
    bands: [
      { label: "Boost Mid", freq: 1500, gain: 4.0, q: 1.0 },
      { label: "Air", freq: 8000, gain: 2.5, q: 0.8 },
    ],
  },
  {
    id: "rock",
    title: "Rock",
    subtitle: "ดัน punch และ attack ในย่าน low และ upper-mid.",
    accent: "#F97316",
    profileLabel: "Punch + Attack",
    graphImage: "/eq-graphs/rock.png",
    bands: [
      { label: "Low Punch", freq: 120, gain: 3.5, q: 0.9 },
      { label: "Presence", freq: 3000, gain: 3.0, q: 1.0 },
    ],
  },
  {
    id: "trap",
    title: "Trap",
    subtitle: "เน้น sub ให้หนัก พร้อม top-end ที่สว่างและมี snap.",
    accent: "#EF4444",
    profileLabel: "Sub + Snap",
    graphImage: "/eq-graphs/trap.png",
    bands: [
      { label: "Sub", freq: 60, gain: 5.0, q: 1.2 },
      { label: "Snap", freq: 8000, gain: 3.5, q: 0.9 },
    ],
  },
  {
    id: "country",
    title: "Country",
    subtitle: "เติม low-mid ให้มี body และเพิ่ม upper clarity ที่สะอาด.",
    accent: "#FACC15",
    profileLabel: "Body + Clarity",
    graphImage: "/eq-graphs/country.png",
    bands: [
      { label: "Body", freq: 250, gain: 2.5, q: 0.9 },
      { label: "Clarity", freq: 4000, gain: 3.0, q: 0.9 },
    ],
  },
  {
    id: "soul",
    title: "Soul",
    subtitle: "คง low ให้อุ่น และให้ high texture ที่ silky.",
    accent: "#34D399",
    profileLabel: "Warm + Silk",
    graphImage: "/eq-graphs/soul.png",
    bands: [
      { label: "Warm", freq: 180, gain: 3.0, q: 1.0 },
      { label: "Silk", freq: 6000, gain: 2.5, q: 0.8 },
    ],
  },
];

function formatFreq(freq: number): string {
  return freq >= 1000 ? `${(freq / 1000).toFixed(freq % 1000 === 0 ? 0 : 1)}kHz` : `${freq}Hz`;
}

// การ์ดอ้างอิงแบบโต้ตอบได้สำหรับเลือกดู preset ของแต่ละ genre
export default function GenreEqCards() {
  const [selectedId, setSelectedId] = useState<GenreId>("pop");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // ปิด dropdown เมื่อผู้ใช้คลิกนอกพื้นที่ของเมนู
    function onClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  const selected = useMemo(
    () => GENRE_PRESETS.find((preset) => preset.id === selectedId) ?? GENRE_PRESETS[0],
    [selectedId]
  );

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2">
        <p className="text-sm uppercase tracking-[0.22em] text-[#A78BFA]">EQ Profiles</p>
        <h2 className="text-2xl font-bold md:text-3xl">EQ Genre</h2>
        <p className="max-w-3xl text-sm text-[#EDE9FE]/75 md:text-base">
          เลือก genre จาก dropdown เพื่อดูภาพ EQ graph และรายละเอียด preset.
        </p>
      </div>

      <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A]/88 p-4 shadow-[0_20px_50px_rgba(8,10,24,0.45)] backdrop-blur md:p-5">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-[#22D3EE]/35 bg-[#111827] text-[#22D3EE]">
              <svg className="h-5 w-5" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path d="M3 4H17M3 10H17M3 16H17" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
            </span>
            <div>
              <p className="text-xs uppercase tracking-wide text-[#A78BFA]">Genre ที่เลือก</p>
              <p className="text-lg font-semibold text-[#EDE9FE]">
                {selected.title}
                <span className="ml-2 rounded-full border border-[#5B21B6]/45 bg-[#111827] px-2 py-0.5 text-xs text-[#C4B5FD]">
                  {selected.profileLabel}
                </span>
              </p>
            </div>
          </div>

          <div ref={dropdownRef} className="relative w-full md:w-72">
            <button
              type="button"
              aria-expanded={isOpen}
              aria-haspopup="listbox"
              onClick={() => setIsOpen((prev) => !prev)}
              className="flex w-full items-center justify-between rounded-xl border border-[#22D3EE]/35 bg-[#111827] px-3 py-2.5 text-left text-sm font-semibold text-[#EDE9FE] transition hover:border-[#22D3EE]/65 hover:bg-[#0B1021]"
            >
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: selected.accent }} />
                {selected.title}
              </span>
              <svg className={`h-4 w-4 text-[#22D3EE] transition-transform ${isOpen ? "rotate-180" : ""}`} viewBox="0 0 20 20" fill="none" aria-hidden="true">
                <path d="M5 8L10 13L15 8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>

            {isOpen && (
              <ul
                role="listbox"
                className="absolute z-20 mt-2 max-h-64 w-full overflow-y-auto rounded-xl border border-[#5B21B6]/35 bg-[#0B1021] p-1.5 shadow-[0_16px_36px_rgba(6,9,22,0.75)]"
              >
                {GENRE_PRESETS.map((preset) => (
                  <li key={preset.id}>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedId(preset.id);
                        setIsOpen(false);
                      }}
                      className={`flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-sm transition ${
                        selectedId === preset.id ? "bg-[#5B21B6]/28 text-[#EDE9FE]" : "text-[#CBD5E1] hover:bg-[#111827]"
                      }`}
                    >
                      <span className="inline-flex items-center gap-2">
                        <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: preset.accent }} />
                        {preset.title}
                      </span>
                      <span className="text-[11px] text-[#94A3B8]">{preset.profileLabel}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="mb-3 rounded-xl border border-[#5B21B6]/28 bg-[#0B1021]/70 p-3 text-sm text-[#C4B5FD]">{selected.subtitle}</div>

        <div className="mx-auto max-w-4xl overflow-hidden rounded-xl border border-[#5B21B6]/28 bg-[#070B18]">
          <Image
            src={selected.graphImage}
            alt={`${selected.title} EQ กราฟ`}
            width={1200}
            height={420}
            sizes="(max-width: 768px) 100vw, 896px"
            className="h-[220px] w-full object-contain md:h-[280px] lg:h-[320px]"
            priority={selected.id === "pop"}
          />
        </div>

        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {selected.bands.map((band) => (
            <div key={`${selected.id}-chip-${band.label}`} className="rounded-lg border border-[#5B21B6]/35 bg-[#111827]/80 px-3 py-2 text-sm">
              <p className="font-semibold text-[#EDE9FE]">{band.label}</p>
              <p className="mt-0.5 text-xs text-[#A78BFA]">
                {formatFreq(band.freq)} / {band.gain > 0 ? `+${band.gain}` : band.gain}dB / Q {band.q}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
