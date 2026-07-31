"use client";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import { toast } from "sonner";
import { CreditCard, Ban, Clock, CheckCircle, XCircle } from "lucide-react";

interface PaymentRecord {
  id: string;
  amount: number;
  currency: string;
  status: string;
  paidAt: string;
}

export default function BillingSection() {
  const router = useRouter();
  const { data: session } = useSession();
  const [payments, setPayments] = useState<PaymentRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [cancelPassword, setCancelPassword] = useState("");
  const [showCancel, setShowCancel] = useState(false);
  const [canceling, setCanceling] = useState(false);

  const currentTier = (session?.user as any)?.tier || "PRO";
  const isPaid = currentTier !== "FREE";
  const hasPassword = (session?.user as { hasPassword?: boolean })?.hasPassword ?? true;

  const handleCancelClick = () => {
    if (!hasPassword) {
      // M19+: OAuth-only ต้อง re-auth ผ่าน provider ก่อนยกเลิก (ไปหน้า confirm-delete)
      router.push("/account/confirm-delete?action=cancel");
      return;
    }
    setShowCancel(true);
  };

  useEffect(() => {
    setLoading(true);
    fetch("/api/subscription/history")
      .then((res) => res.json())
      .then((data) => {
        if (data.payments?.length) setPayments(data.payments);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const formatAmount = (amount: number) => `${(amount / 100).toFixed(2)} THB`;

  const statusStyle = (status: string) => {
    switch (status) {
      case "successful": return "text-[#34D399] bg-[#34D399]/10";
      case "failed": return "text-red-400 bg-red-500/10";
      case "expired": return "text-yellow-400 bg-yellow-500/10";
      default: return "text-[#888] bg-[#1E1B18]";
    }
  };

  const statusIcon = (status: string) => {
    switch (status) {
      case "successful": return <CheckCircle className="w-3 h-3" />;
      case "failed": return <XCircle className="w-3 h-3" />;
      default: return <Clock className="w-3 h-3" />;
    }
  };

  const handleCancel = async () => {
    if (!confirm("Cancel your subscription? You will keep your current tier until the period ends.")) return;
    setCanceling(true);
    try {
      const res = await fetch("/api/subscription/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: cancelPassword || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to cancel subscription");
        return;
      }
      toast.success(data.message);
      setShowCancel(false);
      setCancelPassword("");
    } catch {
      toast.error("Network error");
    } finally {
      setCanceling(false);
    }
  };

  return (
    <div className="space-y-8">
      {isPaid && (
        <div>
          <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Manage Subscription</p>
          <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold">Cancel {currentTier} Plan</p>
                <p className="text-xs text-[#888] mt-0.5">
                  You will keep {currentTier} features until the current period ends, then downgrade to FREE.
                </p>
              </div>
              {!showCancel ? (
                <button
                  onClick={handleCancelClick}
                  className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-400 text-xs font-semibold rounded-lg hover:bg-red-500/20 transition cursor-pointer shrink-0"
                >
                  <Ban className="w-4 h-4" />
                  Cancel
                </button>
              ) : (
                <div className="space-y-2 shrink-0">
                  <input
                    type="password"
                    value={cancelPassword}
                    onChange={(e) => setCancelPassword(e.target.value)}
                    placeholder="Enter password to confirm"
                    className="w-48 bg-[#0D0B0A] border border-red-500/30 rounded-lg px-3 py-2 text-xs text-[#F5F0EB] focus:outline-none focus:border-red-400 transition"
                  />
                  <div className="flex gap-2">
                    <button onClick={handleCancel} disabled={canceling} className="px-3 py-1.5 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg transition cursor-pointer">
                      {canceling ? "..." : "Confirm"}
                    </button>
                    <button onClick={() => { setShowCancel(false); setCancelPassword(""); }} className="px-3 py-1.5 bg-[#1E1B18] text-[#888] text-xs rounded-lg hover:text-white transition cursor-pointer">
                      Back
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Payment History</p>
        {loading ? (
          <div className="text-[#8E8E8E] text-sm py-4">กำลังโหลดประวัติการชำระเงิน...</div>
        ) : payments.length === 0 ? (
          <div className="text-center py-8">
            <CreditCard className="w-10 h-10 text-[#444] mx-auto mb-2" />
            <p className="text-[#888] text-sm">ยังไม่มีประวัติการชำระเงิน</p>
          </div>
        ) : (
          <div className="bg-[#1E1B18] border border-[#222] rounded-xl overflow-hidden">
            <table className="w-full text-left text-xs">
              <thead className="bg-[#111] text-[#888] uppercase">
                <tr>
                  <th className="py-3 px-4 font-medium">Date</th>
                  <th className="py-3 px-4 font-medium">Amount</th>
                  <th className="py-3 px-4 font-medium text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#222]">
                {payments.map((payment) => (
                  <tr key={payment.id} className="hover:bg-[#111] transition-colors">
                    <td className="py-3 px-4 text-[#CCC]">
                      {new Date(payment.paidAt).toLocaleDateString("th-TH", {
                        year: "numeric", month: "short", day: "numeric",
                      })}
                    </td>
                    <td className="py-3 px-4 text-[#CCC] font-mono">{formatAmount(payment.amount)}</td>
                    <td className="py-3 px-4 text-right">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium ${statusStyle(payment.status)}`}>
                        {statusIcon(payment.status)}
                        {payment.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
