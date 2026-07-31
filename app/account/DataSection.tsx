"use client";
import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { signOut } from "next-auth/react";
import { toast } from "sonner";
import { Download, AlertTriangle, Trash2 } from "lucide-react";

interface DataSectionProps {
  hasPassword: boolean;
}

export default function DataSection({ hasPassword }: DataSectionProps) {
  const router = useRouter();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletePassword, setDeletePassword] = useState("");
  const [deleting, setDeleting] = useState(false);

  const handleDeleteClick = () => {
    if (!hasPassword) {
      // M19+: OAuth-only ต้อง re-auth ผ่าน provider ก่อนลบ (ไปหน้า confirm-delete)
      router.push("/account/confirm-delete?action=delete");
      return;
    }
    setShowDeleteModal(true);
  };

  const handleExport = () => {
    const a = document.createElement("a");
    a.href = "/api/account/export";
    a.download = `harmoniq-export-${new Date().toISOString().split("T")[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast.success("Exporting your data...");
  };

  const handleDelete = async () => {
    if (!window.confirm("This will permanently delete your account and all data. Are you sure?")) return;
    setDeleting(true);
    try {
      const res = await fetch("/api/account", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password: deletePassword || undefined }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || "Failed to delete account");
        return;
      }
      toast.success("Account deleted");
      signOut({ callbackUrl: "/" });
    } catch {
      toast.error("Network error");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Export Section */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Export Data</p>
        <div className="p-4 rounded-xl bg-[#1E1B18] border border-[#222] flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold">Download Your Data</p>
            <p className="text-xs text-[#666]">Export profile, preferences, history, and usage data as JSON.</p>
          </div>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-4 py-2 bg-[#F97316]/10 text-[#F97316] text-xs font-semibold rounded-lg hover:bg-[#F97316]/20 transition cursor-pointer"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Delete Account Section */}
      <div>
        <p className="text-xs font-semibold text-[#8E8E8E] uppercase tracking-wider mb-3">Danger Zone</p>
        <div className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
          <div className="flex items-start gap-3 mb-4">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-red-400">Delete Account</p>
              <p className="text-xs text-[#888] mt-1">
                Permanently delete your account, all data, and project history. This action cannot be undone.
              </p>
            </div>
          </div>
          {!showDeleteModal ? (
            <button
              onClick={handleDeleteClick}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-400 text-xs font-semibold rounded-lg hover:bg-red-500/20 transition cursor-pointer"
            >
              <Trash2 className="w-4 h-4" />
              Delete My Account
            </button>
          ) : (
            <div className="space-y-3 border-t border-red-500/10 pt-4">
              {hasPassword && (
                <div>
                  <label className="block text-xs text-[#888] mb-1">Enter your password to confirm:</label>
                  <input
                    type="password"
                    value={deletePassword}
                    onChange={(e) => setDeletePassword(e.target.value)}
                    className="w-full max-w-xs bg-[#0D0B0A] border border-red-500/30 rounded-lg px-3 py-2 text-sm text-[#F5F0EB] focus:outline-none focus:border-red-400 transition"
                  />
                </div>
              )}
              <div className="flex items-center gap-3">
                <button
                  onClick={handleDelete}
                  disabled={deleting}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white text-xs font-bold rounded-lg transition cursor-pointer"
                >
                  {deleting ? "Deleting..." : "Confirm Delete"}
                </button>
                <button
                  onClick={() => { setShowDeleteModal(false); setDeletePassword(""); }}
                  className="px-4 py-2 bg-[#1E1B18] text-[#888] text-xs rounded-lg hover:text-white transition cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
