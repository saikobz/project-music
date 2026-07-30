"use client";
import React, { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { Sun, Moon, Globe, Bell, BellOff } from "lucide-react";

interface PreferencesSectionProps {
  preferences: { theme: string; language: string; emailNotifications: boolean };
  onUpdated: (prefs: { theme: string; language: string; emailNotifications: boolean }) => void;
}

export default function PreferencesSection({ preferences, onUpdated }: PreferencesSectionProps) {
  const [theme, setTheme] = useState(preferences.theme);
  const [language, setLanguage] = useState(preferences.language);
  const [emailNotifications, setEmailNotifications] = useState(preferences.emailNotifications);
  const [saving, setSaving] = useState<string | null>(null);

  // Apply theme on mount
  useEffect(() => {
    const stored = localStorage.getItem("harmoniq-theme");
    if (stored) {
      document.documentElement.classList.toggle("dark", stored === "DARK");
      setTheme(stored);
    }
  }, []);

  const savePreference = useCallback(async (key: string, value: any) => {
    setSaving(key);
    try {
      const res = await fetch("/api/account/preferences", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to save preference");
        return;
      }
      onUpdated(data.preferences);
    } catch {
      toast.error("Network error");
    } finally {
      setSaving(null);
    }
  }, [onUpdated]);

  const handleThemeChange = (newTheme: string) => {
    setTheme(newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "DARK");
    localStorage.setItem("harmoniq-theme", newTheme);
    savePreference("theme", newTheme);
  };

  const handleLanguageChange = (newLanguage: string) => {
    setLanguage(newLanguage);
    savePreference("language", newLanguage);
    toast.success(`Language preference saved (UI translations coming soon)`);
  };

  const handleNotificationChange = (value: boolean) => {
    setEmailNotifications(value);
    savePreference("emailNotifications", value);
  };

  return (
    <div className="space-y-8">
      {/* Theme */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Theme</p>
        <div className="flex gap-3">
          <button
            onClick={() => handleThemeChange("DARK")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              theme === "DARK"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Moon className="w-5 h-5" />
            <span className="text-sm font-semibold">Dark</span>
          </button>
          <button
            onClick={() => handleThemeChange("LIGHT")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              theme === "LIGHT"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Sun className="w-5 h-5" />
            <span className="text-sm font-semibold">Light</span>
          </button>
        </div>
      </div>

      {/* Language */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Language</p>
        <div className="flex gap-3">
          <button
            onClick={() => handleLanguageChange("TH")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              language === "TH"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Globe className="w-5 h-5" />
            <span className="text-sm font-semibold">ไทย</span>
          </button>
          <button
            onClick={() => handleLanguageChange("EN")}
            className={`flex items-center gap-2 px-5 py-3 rounded-xl border transition cursor-pointer ${
              language === "EN"
                ? "bg-[#1A1A1A] border-[#34D399] text-[#34D399]"
                : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
            }`}
          >
            <Globe className="w-5 h-5" />
            <span className="text-sm font-semibold">English</span>
          </button>
        </div>
      </div>

      {/* Email Notifications */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Notifications</p>
        <button
          onClick={() => handleNotificationChange(!emailNotifications)}
          className={`flex items-center gap-3 px-5 py-3 rounded-xl border transition cursor-pointer w-full sm:w-auto ${
            emailNotifications
              ? "bg-[#1A1A1A] border-[#34D399]/40 text-[#34D399]"
              : "bg-[#111] border-[#222] text-[#888] hover:border-[#444]"
          }`}
        >
          {emailNotifications ? <Bell className="w-5 h-5" /> : <BellOff className="w-5 h-5" />}
          <div className="text-left">
            <p className="text-sm font-semibold">Email Notifications</p>
            <p className="text-xs text-[#666]">
              {emailNotifications ? "Receive emails when processing completes" : "No email notifications"}
            </p>
          </div>
        </button>
      </div>
    </div>
  );
}
