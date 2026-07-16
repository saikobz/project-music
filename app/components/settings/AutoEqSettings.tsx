import React from "react";

interface AutoEqSettingsProps {
  autoEqModel: string;
  setAutoEqModel: (val: string) => void;
  deltaClampDb: string;
  setDeltaClampDb: (val: string) => void;
  loading: boolean;
  modelOptions: Array<{ value: string; label: string; hint: string }>;
  minDeltaClamp: number;
  maxDeltaClamp: number;
  defaultDeltaClamp: number;
  isValid: boolean;
  warningText: string;
}

export const AutoEqSettings: React.FC<AutoEqSettingsProps> = ({
  autoEqModel,
  setAutoEqModel,
  deltaClampDb,
  setDeltaClampDb,
  loading,
  modelOptions,
  minDeltaClamp,
  maxDeltaClamp,
  defaultDeltaClamp,
  isValid,
  warningText,
}) => {
  const selectedModel = modelOptions.find((m) => m.value === autoEqModel) || modelOptions[0];

  return (
    <div className="space-y-2">
      <div>
        <label className="block text-sm mb-1">Auto-EQ Model</label>
        <select
          value={autoEqModel}
          onChange={(e) => setAutoEqModel(e.target.value)}
          className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
          disabled={loading}
        >
          {modelOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <p className="mt-1 text-xs text-[#A78BFA]">{selectedModel?.hint}</p>
      </div>
      <div className="flex items-center justify-between gap-3">
        <label className="block text-sm mb-1">Delta Clamp (dB)</label>
        <input
          type="number"
          min={minDeltaClamp}
          max={maxDeltaClamp}
          step="0.1"
          value={deltaClampDb}
          onChange={(e) => setDeltaClampDb(e.target.value)}
          className="w-24 rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-right text-[#EDE9FE]"
          disabled={loading}
        />
      </div>
      <input
        type="range"
        min={minDeltaClamp}
        max={maxDeltaClamp}
        step="0.1"
        value={deltaClampDb}
        onChange={(e) => setDeltaClampDb(e.target.value)}
        className="w-full accent-[#22D3EE]"
        disabled={loading}
      />
      <div className="flex justify-between text-xs text-[#A78BFA]">
        <span>{minDeltaClamp} dB</span>
        <span>ค่าเริ่มต้น {defaultDeltaClamp} dB</span>
        <span>{maxDeltaClamp} dB</span>
      </div>
      <div
        className={`rounded-lg border px-3 py-2 text-xs leading-5 ${
          isValid
            ? "border-amber-400/40 bg-amber-500/10 text-amber-100"
            : "border-red-400/50 bg-red-500/10 text-red-100"
        }`}
      >
        {warningText}
      </div>
    </div>
  );
};
