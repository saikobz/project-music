"use client";
import { useState } from "react";

interface CheckoutModalProps {
  isOpen: boolean;
  onClose: () => void;
  tier: "BASIC" | "PRO";
  price: number;
}

export default function CheckoutModal({ isOpen, onClose, tier, price }: CheckoutModalProps) {
  const [paymentMethod, setPaymentMethod] = useState<"PROMPTPAY" | "CREDIT_CARD">("PROMPTPAY");
  const [qrUrl, setQrUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleCheckout = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/subscription/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tier, paymentMethod }),
      });
      const data = await res.json();
      if (data.qrCodeUrl) {
        setQrUrl(data.qrCodeUrl);
      } else if (data.success) {
        alert("Subscription activated successfully!");
        onClose();
      } else {
        alert(data.error || "Payment failed");
      }
    } catch (err) {
      alert("Payment request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-md w-full text-white shadow-2xl">
        <h2 className="text-xl font-bold mb-2">Subscribe to {tier} Tier</h2>
        <p className="text-slate-400 text-sm mb-4">Total Amount: {price} THB/month</p>

        {qrUrl ? (
          <div className="text-center py-4">
            <p className="text-sm font-semibold mb-2">Scan QR Code via Mobile Banking App</p>
            <img src={qrUrl} alt="PromptPay QR Code" className="mx-auto w-64 h-64 rounded-lg bg-white p-2" />
            <button onClick={onClose} className="mt-4 px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white font-semibold">
              Done
            </button>
          </div>
        ) : (
          <div>
            <div className="space-y-2 mb-6">
              <label className="flex items-center space-x-3 p-3 bg-slate-800/80 rounded-lg cursor-pointer hover:bg-slate-800">
                <input
                  type="radio"
                  name="payment"
                  checked={paymentMethod === "PROMPTPAY"}
                  onChange={() => setPaymentMethod("PROMPTPAY")}
                />
                <span>PromptPay QR Code (Thailand)</span>
              </label>
              <label className="flex items-center space-x-3 p-3 bg-slate-800/80 rounded-lg cursor-pointer hover:bg-slate-800">
                <input
                  type="radio"
                  name="payment"
                  checked={paymentMethod === "CREDIT_CARD"}
                  onChange={() => setPaymentMethod("CREDIT_CARD")}
                />
                <span>Credit / Debit Card (Auto-recurring)</span>
              </label>
            </div>
            <div className="flex justify-end space-x-2">
              <button onClick={onClose} className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg">Cancel</button>
              <button
                onClick={handleCheckout}
                disabled={loading}
                className="px-6 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-semibold"
              >
                {loading ? "Processing..." : "Pay Now"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
