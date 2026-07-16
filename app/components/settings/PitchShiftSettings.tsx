import React from "react";

interface PitchShiftSettingsProps {
  pitchSteps: number;
  setPitchSteps: (val: number) => void;
  loading: boolean;
}

export const PitchShiftSettings: React.FC<PitchShiftSettingsProps> = ({
  pitchSteps,
  setPitchSteps,
  loading,
}) => {
  return (
    <div>
      <label className="block text-sm mb-1">ปรับ pitch (half-steps ±)</label>
      <input
        type="number"
        value={pitchSteps}
        onChange={(e) => setPitchSteps(parseFloat(e.target.value) || 0)}
        className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
        disabled={loading}
      />
    </div>
  );
};
