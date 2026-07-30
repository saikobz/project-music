// app/components/studio/AudioToolSelector.tsx
"use client";

import React from "react";
import { AutoEqSettings } from "../settings/AutoEqSettings";
import { CompressorSettings } from "../settings/CompressorSettings";
import { PitchShiftSettings } from "../settings/PitchShiftSettings";

const AUTO_EQ_DELTA_CLAMP_MIN = 0;
const AUTO_EQ_DELTA_CLAMP_MAX = 6;
const AUTO_EQ_DELTA_CLAMP_DEFAULT = 2;
const AUTO_EQ_MODEL_OPTIONS = [
  { value: "cnn-v1", label: "CNN", hint: "โหมดเดิมของโปรเจกต์" },
  { value: "lstm-last", label: "LSTM", hint: "โมเดลใหม่แบบ sequence-aware" },
];

interface AudioToolSelectorProps {
  action: string;
  setAction: (act: string) => void;
  strength: string;
  setStrength: (val: string) => void;
  genre: string;
  setGenre: (val: string) => void;
  autoEqModel: string;
  setAutoEqModel: (val: string) => void;
  deltaClampDb: string;
  setDeltaClampDb: (val: string) => void;
  compThreshold: string;
  setCompThreshold: (val: string) => void;
  compRatio: string;
  setCompRatio: (val: string) => void;
  compAttack: string;
  setCompAttack: (val: string) => void;
  compRelease: string;
  setCompRelease: (val: string) => void;
  compKnee: string;
  setCompKnee: (val: string) => void;
  compMakeupGain: string;
  setCompMakeupGain: (val: string) => void;
  compDryWet: string;
  setCompDryWet: (val: string) => void;
  compOutputCeiling: string;
  setCompOutputCeiling: (val: string) => void;
  pitchSteps: number;
  setPitchSteps: (val: number) => void;
  loading: boolean;
}

export function AudioToolSelector({
  action,
  setAction,
  strength,
  setStrength,
  genre,
  setGenre,
  autoEqModel,
  setAutoEqModel,
  deltaClampDb,
  setDeltaClampDb,
  compThreshold,
  setCompThreshold,
  compRatio,
  setCompRatio,
  compAttack,
  setCompAttack,
  compRelease,
  setCompRelease,
  compKnee,
  setCompKnee,
  compMakeupGain,
  setCompMakeupGain,
  compDryWet,
  setCompDryWet,
  compOutputCeiling,
  setCompOutputCeiling,
  pitchSteps,
  setPitchSteps,
  loading,
}: AudioToolSelectorProps) {
  const parsedClamp = parseFloat(deltaClampDb);
  const isEqDeltaClampValid = !isNaN(parsedClamp) && parsedClamp >= AUTO_EQ_DELTA_CLAMP_MIN && parsedClamp <= AUTO_EQ_DELTA_CLAMP_MAX;
  const eqDeltaClampWarning = isEqDeltaClampValid
    ? ""
    : `กรุณากรอกค่าเพดานการปรับ EQ ระหว่าง ${AUTO_EQ_DELTA_CLAMP_MIN} ถึง ${AUTO_EQ_DELTA_CLAMP_MAX} dB`;

  return (
    <div className="rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] shadow-xl overflow-hidden">
      {/* Console Header with Status Indicator */}
      <div className="flex items-center gap-2.5 px-4 pt-4 pb-3 border-b border-[#1E1E1E]">
        <span
          className={`w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300 ${
            loading
              ? "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.6)] animate-pulse"
              : action === "separate"
              ? "bg-[#A78BFA] shadow-[0_0_5px_rgba(167,139,250,0.5)]"
              : action === "auto_eq"
              ? "bg-[#22D3EE] shadow-[0_0_5px_rgba(34,211,238,0.5)]"
              : action === "compressor"
              ? "bg-[#E5A93D] shadow-[0_0_5px_rgba(229,169,61,0.5)]"
              : "bg-[#34D399] shadow-[0_0_5px_rgba(52,211,153,0.5)]"
          }`}
        />
        <p className="text-[10px] font-semibold text-[#555555] uppercase tracking-[0.15em]">
          Processing Module
        </p>
        <div className="ml-auto text-[10px] font-mono text-[#333333]">
          {loading ? "RUNNING" : "STANDBY"}
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Tab Selector */}
        <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-[#080808] border border-[#1A1A1A]">
          {([
            {
              value: "separate",
              label: "Stem Separation",
              icon: "⊗",
              activeColor:
                "text-[#A78BFA] border-[#A78BFA]/60 bg-[#A78BFA]/10 shadow-[0_0_12px_rgba(167,139,250,0.15)]",
            },
            {
              value: "auto_eq",
              label: "Auto EQ (AI)",
              icon: "⟿",
              activeColor:
                "text-[#22D3EE] border-[#22D3EE]/60 bg-[#22D3EE]/10 shadow-[0_0_12px_rgba(34,211,238,0.15)]",
            },
            {
              value: "compressor",
              label: "Compressor",
              icon: "◉",
              activeColor:
                "text-[#E5A93D] border-[#E5A93D]/60 bg-[#E5A93D]/10 shadow-[0_0_12px_rgba(229,169,61,0.15)]",
            },
            {
              value: "pitch_shift",
              label: "Pitch Shift",
              icon: "♯",
              activeColor:
                "text-[#34D399] border-[#34D399]/60 bg-[#34D399]/10 shadow-[0_0_12px_rgba(52,211,153,0.15)]",
            },
          ] as const).map((item) => (
            <button
              key={item.value}
              onClick={() => setAction(item.value)}
              disabled={loading}
              className={`relative flex flex-col items-center gap-0.5 rounded-md px-2 py-2.5 text-xs font-medium border transition-all duration-200 cursor-pointer ${
                action === item.value
                  ? item.activeColor
                  : "text-[#444444] border-transparent bg-transparent hover:text-[#888888] hover:bg-[#111111]"
              } disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              <span
                className={`text-base leading-none transition-all duration-200 ${
                  action === item.value ? "opacity-100" : "opacity-40"
                }`}
              >
                {item.icon}
              </span>
              <span className="leading-tight text-center">{item.label}</span>
            </button>
          ))}
        </div>

        {/* Info card for Stem Separation */}
        {action === "separate" && (
          <div className="rounded-lg border border-[#A78BFA]/20 bg-[#A78BFA]/5 p-3 space-y-2">
            <p className="text-xs text-[#A78BFA] font-medium">Stem Separation</p>
            <p className="text-[11px] text-[#666666] leading-relaxed">
              แยกไฟล์เสียงออกเป็น 4 แทร็กอิสระ — Vocals, Drums, Bass, Other — พร้อมเล่นและ Mix ได้ทันที
            </p>
            <div className="grid grid-cols-4 gap-1 pt-1">
              {["Vocals", "Drums", "Bass", "Other"].map((s) => (
                <div
                  key={s}
                  className="rounded bg-[#A78BFA]/10 border border-[#A78BFA]/20 py-1 text-center text-[10px] text-[#A78BFA]/80"
                >
                  {s}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Genre Profile selector for Auto-EQ or Compressor */}
        {(action === "auto_eq" || action === "compressor") && (
          <div>
            <label className="block text-[10px] font-semibold text-[#555555] uppercase tracking-[0.12em] mb-1.5">
              Genre Profile
            </label>
            <select
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
              className={`w-full rounded-lg bg-[#080808] border p-2.5 text-[#C8C8C8] text-sm focus:outline-none transition ${
                action === "auto_eq"
                  ? "border-[#22D3EE]/30 focus:border-[#22D3EE]/70"
                  : "border-[#E5A93D]/30 focus:border-[#E5A93D]/70"
              }`}
              disabled={loading}
            >
              <option value="pop">Pop</option>
              <option value="rock">Rock</option>
              <option value="trap">Trap</option>
              <option value="country">Country</option>
              <option value="soul">Soul</option>
            </select>
          </div>
        )}

        {/* Auto EQ Controls */}
        {action === "auto_eq" && (
          <div className="rounded-lg border border-[#22D3EE]/20 bg-[#22D3EE]/5 p-3">
            <AutoEqSettings
              autoEqModel={autoEqModel}
              setAutoEqModel={setAutoEqModel}
              deltaClampDb={deltaClampDb}
              setDeltaClampDb={setDeltaClampDb}
              loading={loading}
              modelOptions={AUTO_EQ_MODEL_OPTIONS}
              minDeltaClamp={AUTO_EQ_DELTA_CLAMP_MIN}
              maxDeltaClamp={AUTO_EQ_DELTA_CLAMP_MAX}
              defaultDeltaClamp={AUTO_EQ_DELTA_CLAMP_DEFAULT}
              isValid={isEqDeltaClampValid}
              warningText={eqDeltaClampWarning}
            />
          </div>
        )}

        {/* Compressor Controls */}
        {action === "compressor" && (
          <div className="rounded-lg border border-[#E5A93D]/20 bg-[#E5A93D]/5 p-3">
            <CompressorSettings
              strength={strength}
              setStrength={setStrength}
              compThreshold={compThreshold}
              setCompThreshold={setCompThreshold}
              compRatio={compRatio}
              setCompRatio={setCompRatio}
              compAttack={compAttack}
              setCompAttack={setCompAttack}
              compRelease={compRelease}
              setCompRelease={setCompRelease}
              compKnee={compKnee}
              setCompKnee={setCompKnee}
              compMakeupGain={compMakeupGain}
              setCompMakeupGain={setCompMakeupGain}
              compDryWet={compDryWet}
              setCompDryWet={setCompDryWet}
              compOutputCeiling={compOutputCeiling}
              setCompOutputCeiling={setCompOutputCeiling}
              loading={loading}
            />
          </div>
        )}

        {/* Pitch Shift Controls */}
        {action === "pitch_shift" && (
          <div className="rounded-lg border border-[#34D399]/20 bg-[#34D399]/5 p-3">
            <PitchShiftSettings
              pitchSteps={pitchSteps}
              setPitchSteps={setPitchSteps}
              loading={loading}
            />
          </div>
        )}
      </div>
    </div>
  );
}
