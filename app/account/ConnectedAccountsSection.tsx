"use client";
import React, { useEffect, useState } from "react";
import { signIn } from "next-auth/react";
import { toast } from "sonner";
import { Link2, Unlink, Mail, Globe } from "lucide-react";

interface Provider {
  id: string;
  name: string;
  icon: string;
  linked: boolean;
}

const PROVIDER_ICONS: Record<string, React.ReactNode> = {
  google: <Globe className="w-5 h-5" />,
  facebook: (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M24 12.073c0-6.627-5.373-12-12-12S0 5.446 0 12.073c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
    </svg>
  ),
  line: (
    <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm3.5 13.5c-.28.28-.72.28-1 0l-2.5-2.5-2.5 2.5c-.28.28-.72.28-1 0s-.28-.72 0-1l3-3c.28-.28.72-.28 1 0l3 3c.28.28.28.72 0 1z"/>
    </svg>
  ),
  mail: <Mail className="w-5 h-5" />,
};

export default function ConnectedAccountsSection() {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/account/providers")
      .then((res) => res.json())
      .then((data) => setProviders(data.providers || []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleLink = async (providerId: string) => {
    if (providerId === "credentials") return;
    try {
      await signIn(providerId, { redirect: false });
      const res = await fetch("/api/account/providers");
      const data = await res.json();
      setProviders(data.providers || []);
      toast.success(`Connected to ${providerId}`);
    } catch {
      toast.error("Failed to connect account");
    }
  };

  const handleUnlink = async (providerId: string) => {
    if (providerId === "credentials") return;
    if (!window.confirm(`Disconnect ${providerId}? You may lose access if you have no other login method.`)) return;
    try {
      const res = await fetch(`/api/account/providers`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: providerId }),
      });
      if (!res.ok) throw new Error();
      setProviders((prev) =>
        prev.map((p) => (p.id === providerId ? { ...p, linked: false } : p))
      );
      toast.success(`Disconnected ${providerId}`);
    } catch {
      toast.error("Failed to disconnect account");
    }
  };

  if (loading) {
    return <div className="text-[#8E8E8E] text-sm py-4">Loading connected accounts...</div>;
  }

  return (
    <div className="space-y-4">
      {providers.map((provider) => (
        <div
          key={provider.id}
          className="flex items-center justify-between p-4 rounded-xl bg-[#1E1B18] border border-[#2C2824]"
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-[#2C2824] flex items-center justify-center text-[#8E8E8E]">
              {PROVIDER_ICONS[provider.icon] || PROVIDER_ICONS.mail}
            </div>
            <div>
              <p className="text-sm font-semibold">{provider.name}</p>
              <p className="text-xs text-[#5C5854]">
                {provider.linked ? "Connected" : "Not connected"}
              </p>
            </div>
          </div>
          {provider.id !== "credentials" && (
            <button
              onClick={() => (provider.linked ? handleUnlink(provider.id) : handleLink(provider.id))}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition cursor-pointer ${
                provider.linked
                  ? "bg-red-500/10 text-red-400 hover:bg-red-500/20"
                  : "bg-[#F97316]/10 text-[#F97316] hover:bg-[#F97316]/20"
              }`}
            >
              {provider.linked ? (
                <><Unlink className="w-3.5 h-3.5" /> Disconnect</>
              ) : (
                <><Link2 className="w-3.5 h-3.5" /> Connect</>
              )}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
