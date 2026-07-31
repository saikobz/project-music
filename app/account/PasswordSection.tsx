"use client";
import React, { useState } from "react";
import { toast } from "sonner";
import { KeyRound, Eye, EyeOff } from "lucide-react";

interface PasswordSectionProps {
  hasPassword: boolean;
}

export default function PasswordSection({ hasPassword }: PasswordSectionProps) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [show, setShow] = useState(false);
  const [saving, setSaving] = useState(false);

  if (!hasPassword) {
    return (
      <div className="text-center py-8 space-y-3">
        <KeyRound className="w-10 h-10 text-[#444] mx-auto" />
        <p className="text-[#8E8E8E] text-sm">You signed up with OAuth and don&apos;t have a password yet.</p>
        <p className="text-[#666] text-xs">Password login is not available for OAuth accounts.</p>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }
    if (newPassword.length < 6) {
      toast.error("Password must be at least 6 characters");
      return;
    }
    setSaving(true);
    try {
      const res = await fetch("/api/account/password", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ currentPassword, newPassword }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to change password");
        return;
      }
      toast.success("Password changed successfully");
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-md space-y-5">
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Current Password
        </label>
        <div className="relative">
          <KeyRound className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5C5854]" />
          <input
            type={show ? "text" : "password"}
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            required
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg pl-10 pr-10 py-2.5 text-sm text-[#F5F0EB] focus:outline-none focus:border-[#F97316] transition"
          />
          <button type="button" onClick={() => setShow(!show)} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#5C5854] hover:text-white cursor-pointer">
            {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          New Password
        </label>
        <input
          type={show ? "text" : "password"}
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          required
          minLength={6}
          className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg px-3 py-2.5 text-sm text-[#F5F0EB] focus:outline-none focus:border-[#F97316] transition"
        />
      </div>

      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Confirm New Password
        </label>
        <input
          type={show ? "text" : "password"}
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          required
          minLength={6}
          className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg px-3 py-2.5 text-sm text-[#F5F0EB] focus:outline-none focus:border-[#F97316] transition"
        />
      </div>

      <button
        type="submit"
        disabled={saving}
        className="px-6 py-2.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] disabled:opacity-50 text-white text-sm font-bold rounded-lg transition cursor-pointer"
      >
        {saving ? "Changing..." : "Change Password"}
      </button>
    </form>
  );
}
