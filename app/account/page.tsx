"use client";
import React, { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { User, KeyRound, Link2, Settings, Trash2, CreditCard } from "lucide-react";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";
import ProfileSection from "./ProfileSection";
import PasswordSection from "./PasswordSection";
import ConnectedAccountsSection from "./ConnectedAccountsSection";
import PreferencesSection from "./PreferencesSection";
import DataSection from "./DataSection";
import BillingSection from "./BillingSection";

type AccountTab = "profile" | "password" | "accounts" | "preferences" | "data" | "billing";

const TABS: { id: AccountTab; label: string; icon: React.ReactNode }[] = [
  { id: "profile", label: "Profile", icon: <User className="w-4 h-4" /> },
  { id: "password", label: "Password", icon: <KeyRound className="w-4 h-4" /> },
  { id: "accounts", label: "Connected Accounts", icon: <Link2 className="w-4 h-4" /> },
  { id: "preferences", label: "Preferences", icon: <Settings className="w-4 h-4" /> },
  { id: "data", label: "Data", icon: <Trash2 className="w-4 h-4" /> },
  { id: "billing", label: "Billing", icon: <CreditCard className="w-4 h-4" /> },
];

export default function AccountPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<AccountTab>("profile");

  const fetchAccount = useCallback(() => {
    setLoading(true);
    fetch("/api/account")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => { fetchAccount(); }, [fetchAccount]);

  const handleProfileUpdated = (updates: { name: string | null; email: string; image: string | null }) => {
    setData((prev: any) => ({
      ...prev,
      user: { ...prev.user, ...updates },
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center text-[#8E8E8E]">Loading account details...</div>
        <Footer />
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center space-y-4">
          <p className="text-red-400 font-medium">Please sign in to view your account details.</p>
          <Link href="/api/auth/signin" className="inline-block px-6 py-2 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white font-bold rounded-lg transition">
            Sign In
          </Link>
        </div>
        <Footer />
      </div>
    );
  }

  const { user, subscription, quota } = data;
  const used = quota.usedCount || 0;
  const max = quota.monthlyQuota;
  const isUnlimited = max === -1;
  const percent = isUnlimited ? 0 : Math.min(100, Math.round((used / max) * 100));

  return (
    <div className="min-h-screen bg-[#0D0B0A] text-[#F5F0EB] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow mx-auto w-full max-w-4xl px-4 py-12 space-y-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Account Settings</h1>
          <p className="text-[#8E8E8E] text-sm mt-1">Manage your profile, connected accounts, and preferences.</p>
        </div>

        {/* Tab Bar */}
        <div className="flex flex-wrap gap-1 border-b border-[#222] pb-0">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium rounded-t-lg transition cursor-pointer whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-[#111] text-[#F97316] border border-b-0 border-[#222] -mb-[1px]"
                  : "text-[#888] hover:text-white hover:bg-[#111]"
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Subscription & Quota Card (always visible) */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider">Current Subscription</p>
              <h3 className="text-xl font-extrabold text-[#34D399] mt-0.5">{subscription.tier} PLAN</h3>
            </div>
            <Link href="/pricing" className="px-4 py-2 bg-gradient-to-br from-[#F97316] to-[#EA580C] hover:from-[#FB923C] hover:to-[#F97316] text-white text-xs font-bold rounded-lg transition">
              Change Plan
            </Link>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-[#CCCCCC]">Monthly Song Processing Quota</span>
            <span className="font-bold text-[#34D399]">{isUnlimited ? "Unlimited" : `${used} / ${max} songs used`}</span>
          </div>
          {!isUnlimited && (
            <div className="w-full bg-[#222222] rounded-full h-2.5 overflow-hidden">
              <div className="bg-[#34D399] h-2.5 rounded-full transition-all duration-300" style={{ width: `${percent}%` }}></div>
            </div>
          )}
        </div>

        {/* Tab Content */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6">
          {activeTab === "profile" && (
            <ProfileSection user={user} onUpdated={handleProfileUpdated} />
          )}
          {activeTab === "password" && (
            <PasswordSection hasPassword={data.user.hasPassword} />
          )}
          {activeTab === "accounts" && <ConnectedAccountsSection />}
          {activeTab === "preferences" && (
            <PreferencesSection
              preferences={data.preferences || { theme: "DARK", language: "TH", emailNotifications: true }}
              onUpdated={(prefs) => setData((prev: any) => ({ ...prev, preferences: prefs }))}
            />
          )}
          {activeTab === "data" && <DataSection hasPassword={data.user.hasPassword} />}
          {activeTab === "billing" && <BillingSection />}
        </div>
      </main>

      <Footer />
    </div>
  );
}
