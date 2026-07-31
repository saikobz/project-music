import React from "react";

// C12: รวมพารามิเตอร์ compressor เป็น object เดียว (เดิม 17 props แยกกัน)
export interface CompressorParams {
  strength: string;
  threshold: string;
  ratio: string;
  attack: string;
  release: string;
  knee: string;
  makeupGain: string;
  dryWet: string;
  outputCeiling: string;
}

export const DEFAULT_COMPRESSOR_PARAMS: CompressorParams = {
  strength: "medium",
  threshold: "",
  ratio: "",
  attack: "",
  release: "",
  knee: "6",
  makeupGain: "0",
  dryWet: "100",
  outputCeiling: "",
};

interface CompressorSettingsProps {
  params: CompressorParams;
  onChange: (patch: Partial<CompressorParams>) => void;
  loading: boolean;
}

// กำหนด field ทั้งหมดใน config เดียว (DRY: เดิม markup ซ้ำ 8 ชุด)
const FIELDS: { key: keyof CompressorParams; label: string; step?: string; placeholder?: string }[] = [
  { key: "threshold", label: "Threshold (dBFS)", step: "0.1", placeholder: "Preset" },
  { key: "ratio", label: "Ratio", step: "0.1", placeholder: "Preset" },
  { key: "attack", label: "Attack (ms)", step: "0.1", placeholder: "Preset" },
  { key: "release", label: "Release (ms)", step: "0.1", placeholder: "Preset" },
  { key: "knee", label: "Knee (dB)", step: "0.1" },
  { key: "makeupGain", label: "Makeup Gain (dB)", step: "0.1" },
  { key: "dryWet", label: "Dry/Wet (%)", step: "1" },
  { key: "outputCeiling", label: "Output Ceiling (dBFS)", step: "0.1", placeholder: "Off" },
];

export const CompressorSettings: React.FC<CompressorSettingsProps> = ({
  params,
  onChange,
  loading,
}) => {
  return (
    <div className="space-y-3">
      <div>
        <label className="block text-sm mb-1">Strength</label>
        <select
          value={params.strength}
          onChange={(e) => onChange({ strength: e.target.value })}
          className="w-full rounded-lg bg-[#0D0B0A] border border-[#2C2824] p-2.5 text-[#F5F0EB] focus:border-[#E5A93D] focus:outline-none transition"
          disabled={loading}
        >
          <option value="soft">Soft</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        {FIELDS.map(({ key, label, step, placeholder }) => (
          <div key={key}>
            <label className="block text-xs mb-1">{label}</label>
            <input
              type="number"
              step={step}
              placeholder={placeholder}
              value={params[key]}
              onChange={(e) => onChange({ [key]: e.target.value } as Partial<CompressorParams>)}
              className="w-full rounded-lg bg-[#0D0B0A] border border-[#2C2824] p-2.5 text-[#F5F0EB] focus:border-[#E5A93D] focus:outline-none transition"
              disabled={loading}
            />
          </div>
        ))}
      </div>
    </div>
  );
};
