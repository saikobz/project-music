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
        className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
        disabled={loading}
      />
    </div>
  );
};
