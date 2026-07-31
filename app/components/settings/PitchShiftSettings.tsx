import React, { useEffect, useState } from "react";

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
  // เก็บค่าดิบที่กำลังพิมพ์ไว้ (M12: เดิม parseFloat("-") -> NaN -> fallback 0
  // ทำให้พิมพ์เครื่องหมายลบไม่ได้เลย) — จะแปลงเป็นตัวเลขเมื่อพิมพ์สมบูรณ์เท่านั้น
  const [rawSteps, setRawSteps] = useState<string>(String(pitchSteps));

  // sync เมื่อค่าจากภายนอกเปลี่ยน (เช่น parent reset หรือ clamp)
  useEffect(() => {
    setRawSteps(String(pitchSteps));
  }, [pitchSteps]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setRawSteps(raw);

    // ยังพิมพ์ไม่เสร็จ (ว่าง / ติด "-" หรือ "." อยู่) -> เก็บ raw ไว้ก่อน ไม่แปลง
    if (raw === "" || raw === "-" || raw === "." || raw === "-." || raw === "+") {
      return;
    }
    const num = Number(raw);
    if (Number.isFinite(num)) {
      setPitchSteps(num);
    }
  };

  return (
    <div>
      <label className="block text-sm mb-1">ปรับ pitch (half-steps ±)</label>
      <input
        type="number"
        value={rawSteps}
        onChange={handleChange}
        className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
        disabled={loading}
        data-testid="pitch-input"
      />
    </div>
  );
};
