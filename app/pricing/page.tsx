"use client";
import { useState } from "react";
import CheckoutModal from "../components/CheckoutModal";

export default function PricingPage() {
  const [selectedTier, setSelectedTier] = useState<"BASIC" | "PRO" | null>(null);

  return (
    <div className="min-h-screen bg-slate-950 text-white py-16 px-4">
      <div className="max-w-5xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-extrabold mb-4 bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent">
          HarmoniQ Plans & Pricing
        </h1>
        <p className="text-slate-400">Choose the perfect plan for your music separation & mastering needs</p>
      </div>

      <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        {/* Free Plan */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-xl font-bold mb-2">Free</h3>
            <div className="text-3xl font-extrabold mb-4">0 THB</div>
            <ul className="text-slate-400 text-sm space-y-2 mb-6">
              <li>• 1 song / month</li>
              <li>• AutoEQ LSTM Model</li>
              <li>• 🔒 CNN Model Locked</li>
            </ul>
          </div>
          <button className="w-full py-2 bg-slate-800 text-slate-400 rounded-lg cursor-not-allowed">Current Plan</button>
        </div>

        {/* Basic Plan */}
        <div className="bg-slate-900 border border-indigo-500/40 rounded-2xl p-6 flex flex-col justify-between relative">
          <div>
            <h3 className="text-xl font-bold mb-2">Basic</h3>
            <div className="text-3xl font-extrabold mb-4">99 THB<span className="text-sm font-normal text-slate-400">/mo</span></div>
            <ul className="text-slate-400 text-sm space-y-2 mb-6">
              <li>• 15 songs / month</li>
              <li>• AutoEQ LSTM & CNN Models</li>
              <li>• Lossless WAV Export</li>
            </ul>
          </div>
          <button
            onClick={() => setSelectedTier("BASIC")}
            className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 font-semibold text-white rounded-lg transition-colors"
          >
            Upgrade to Basic
          </button>
        </div>

        {/* Pro Plan */}
        <div className="bg-gradient-to-b from-indigo-900/40 to-slate-900 border border-indigo-400 rounded-2xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-xl font-bold mb-2 text-indigo-300">Pro</h3>
            <div className="text-3xl font-extrabold mb-4">299 THB<span className="text-sm font-normal text-slate-400">/mo</span></div>
            <ul className="text-slate-300 text-sm space-y-2 mb-6">
              <li>• Unlimited songs / month</li>
              <li>• All AutoEQ Models (LSTM & CNN)</li>
              <li>• Full AI Auto Mastering</li>
            </ul>
          </div>
          <button
            onClick={() => setSelectedTier("PRO")}
            className="w-full py-2 bg-gradient-to-r from-purple-500 to-indigo-500 hover:opacity-90 font-semibold text-white rounded-lg transition-opacity"
          >
            Upgrade to Pro
          </button>
        </div>
      </div>

      {selectedTier && (
        <CheckoutModal
          isOpen={!!selectedTier}
          onClose={() => setSelectedTier(null)}
          tier={selectedTier}
          price={selectedTier === "PRO" ? 299 : 99}
        />
      )}
    </div>
  );
}
