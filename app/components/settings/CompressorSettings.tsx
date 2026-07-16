import React from "react";

interface CompressorSettingsProps {
  strength: string;
  setStrength: (val: string) => void;
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
  loading: boolean;
}

export const CompressorSettings: React.FC<CompressorSettingsProps> = ({
  strength,
  setStrength,
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
  loading,
}) => {
  return (
    <div className="space-y-3">
      <div>
        <label className="block text-sm mb-1">Strength</label>
        <select
          value={strength}
          onChange={(e) => setStrength(e.target.value)}
          className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
          disabled={loading}
        >
          <option value="soft">Soft</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs mb-1">Threshold (dBFS)</label>
          <input
            type="number"
            step="0.1"
            placeholder="Preset"
            value={compThreshold}
            onChange={(e) => setCompThreshold(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Ratio</label>
          <input
            type="number"
            step="0.1"
            placeholder="Preset"
            value={compRatio}
            onChange={(e) => setCompRatio(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Attack (ms)</label>
          <input
            type="number"
            step="0.1"
            placeholder="Preset"
            value={compAttack}
            onChange={(e) => setCompAttack(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Release (ms)</label>
          <input
            type="number"
            step="0.1"
            placeholder="Preset"
            value={compRelease}
            onChange={(e) => setCompRelease(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Knee (dB)</label>
          <input
            type="number"
            step="0.1"
            value={compKnee}
            onChange={(e) => setCompKnee(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Makeup Gain (dB)</label>
          <input
            type="number"
            step="0.1"
            value={compMakeupGain}
            onChange={(e) => setCompMakeupGain(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Dry/Wet (%)</label>
          <input
            type="number"
            step="1"
            min="0"
            max="100"
            value={compDryWet}
            onChange={(e) => setCompDryWet(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Output Ceiling (dBFS)</label>
          <input
            type="number"
            step="0.1"
            placeholder="Off"
            value={compOutputCeiling}
            onChange={(e) => setCompOutputCeiling(e.target.value)}
            className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
            disabled={loading}
          />
        </div>
      </div>
    </div>
  );
};
