"use client";
import React, { useState } from "react";
import { Eye, EyeOff } from "lucide-react";

type Props = {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  required?: boolean;
};

// Input รหัสผ่านพร้อมปุ่มแสดง/ซ่อน (ใช้ร่วมกันใน Sign In และ Sign Up)
export default function PasswordInput({ value, onChange, placeholder = "••••••••", required = true }: Props) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="relative">
      <input
        type={visible ? "text" : "password"}
        required={required}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg p-2.5 pr-10 text-sm text-white focus:outline-none focus:border-[#F97316] transition"
      />
      <button
        type="button"
        onClick={() => setVisible((v) => !v)}
        aria-label={visible ? "ซ่อนรหัสผ่าน" : "แสดงรหัสผ่าน"}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#5C5854] hover:text-[#A09890] transition cursor-pointer"
      >
        {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
      </button>
    </div>
  );
}
