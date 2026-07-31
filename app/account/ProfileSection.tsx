"use client";
import React, { useState } from "react";
import { toast } from "sonner";
import { User, Mail } from "lucide-react";

interface ProfileSectionProps {
  user: { name: string | null; email: string; image: string | null };
  onUpdated: (updates: { name: string | null; email: string; image: string | null }) => void;
}

export default function ProfileSection({ user, onUpdated }: ProfileSectionProps) {
  const [name, setName] = useState(user.name || "");
  const [email, setEmail] = useState(user.email);
  const [image, setImage] = useState(user.image || "");
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const res = await fetch("/api/account/profile", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: name || null, email, image: image || null }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to update profile");
        return;
      }
      onUpdated(data.user);
      toast.success("Profile updated");
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Avatar */}
      <div className="flex items-center gap-4">
        {image ? (
          <img src={image} alt="Avatar" className="w-16 h-16 rounded-full object-cover border border-[#36322E]" />
        ) : (
          <div className="w-16 h-16 rounded-full bg-[#2C2824] flex items-center justify-center">
            <User className="w-8 h-8 text-[#5C5854]" />
          </div>
        )}
        <div className="flex-1">
          <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
            Avatar URL
          </label>
          <input
            type="url"
            value={image}
            onChange={(e) => setImage(e.target.value)}
            placeholder=""
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg px-3 py-2 text-sm text-[#F5F0EB] placeholder-[#5C5854] focus:outline-none focus:border-[#F97316] transition"
          />
        </div>
      </div>

      {/* Name */}
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Display Name
        </label>
        <div className="relative">
          <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5C5854]" />
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg pl-10 pr-3 py-2.5 text-sm text-[#F5F0EB] placeholder-[#5C5854] focus:outline-none focus:border-[#F97316] transition"
          />
        </div>
      </div>

      {/* Email */}
      <div>
        <label className="block text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-1">
          Email
        </label>
        <div className="relative">
          <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5C5854]" />
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full bg-[#1E1B18] border border-[#36322E] rounded-lg pl-10 pr-3 py-2.5 text-sm text-[#F5F0EB] placeholder-[#5C5854] focus:outline-none focus:border-[#F97316] transition"
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={saving}
        className="px-6 py-2.5 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] disabled:opacity-50 text-white text-sm font-bold rounded-lg transition cursor-pointer"
      >
        {saving ? "Saving..." : "Save Changes"}
      </button>
    </form>
  );
}
