"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Navbar } from "../components/Navbar";
import { Footer } from "../components/Footer";

export default function AccountPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/account")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center text-[#8E8E8E]">Loading account details...</div>
        <Footer />
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-3xl mx-auto py-16 text-center space-y-4">
          <p className="text-red-400 font-medium">Please sign in to view your account details.</p>
          <Link href="/api/auth/signin" className="inline-block px-6 py-2 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] font-bold rounded-lg transition">
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
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow mx-auto w-full max-w-3xl px-4 py-12 space-y-8">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight">Account Dashboard</h1>
          <p className="text-[#8E8E8E] text-sm mt-1">Manage your profile, active subscription, and usage quotas.</p>
        </div>

        {/* Profile Card */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 flex items-center gap-4">
          {user.image ? (
            <img src={user.image} alt={user.name} className="w-16 h-16 rounded-full" />
          ) : (
            <div className="w-16 h-16 rounded-full bg-[#222222] flex items-center justify-center text-xl font-bold text-[#F3F3F3]">
              {user.name ? user.name[0].toUpperCase() : "U"}
            </div>
          )}
          <div>
            <h2 className="text-lg font-bold">{user.name || "HarmoniQ User"}</h2>
            <p className="text-sm text-[#8E8E8E]">{user.email}</p>
          </div>
        </div>

        {/* Subscription & Quota Card */}
        <div className="bg-[#111111] border border-[#222222] rounded-2xl p-6 space-y-6">
          <div className="flex items-center justify-between border-b border-[#222222] pb-4">
            <div>
              <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider">Current Subscription</p>
              <h3 className="text-xl font-extrabold text-[#34D399] mt-0.5">{subscription.tier} PLAN</h3>
            </div>
            <Link
              href="/pricing"
              className="px-4 py-2 bg-[#34D399] hover:bg-[#2cb984] text-[#0A0A0A] text-xs font-bold rounded-lg transition"
            >
              Change Plan
            </Link>
          </div>

          {/* Quota Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-[#CCCCCC]">Monthly Song Processing Quota</span>
              <span className="font-bold text-[#34D399]">
                {isUnlimited ? "Unlimited" : `${used} / ${max} songs used`}
              </span>
            </div>
            {!isUnlimited && (
              <div className="w-full bg-[#222222] rounded-full h-2.5 overflow-hidden">
                <div className="bg-[#34D399] h-2.5 rounded-full transition-all duration-300" style={{ width: `${percent}%` }}></div>
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
