"use client";
import React from "react";
import { signIn } from "next-auth/react";

// SVG Logo ตามแบรนด์ของแต่ละผู้ให้บริการ OAuth
function GoogleIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" aria-hidden="true">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.85-2.22.81-.62z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
    </svg>
  );
}

function FacebookIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" aria-hidden="true">
      <path fill="#1877F2" d="M24 12.073C24 5.405 18.627 0 12 0S0 5.405 0 12.073C0 18.1 4.388 23.094 10.125 24v-8.437H7.078v-3.49h3.047v-2.66c0-3.026 1.792-4.697 4.533-4.697 1.313 0 2.686.236 2.686.236v2.971H15.83c-1.491 0-1.956.93-1.956 1.886v2.264h3.328l-.532 3.49h-2.796V24C19.612 23.094 24 18.1 24 12.073z" />
    </svg>
  );
}

function LineIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" aria-hidden="true">
      <path fill="#00C300" d="M11.365 3.839c-5.57 0-10.085 4.013-10.085 8.962 0 2.778 1.544 5.254 3.964 6.873-.427 1.597-1.321 3.694-2.493 5.055-.47.543-.02 1.295.665 1.112 2.457-.651 4.422-1.844 5.814-2.763.914.204 1.87.318 2.855.318 5.569 0 10.084-4.013 10.084-8.961 0-4.95-4.515-8.963-10.084-8.963zM4.216 12.918v-1.871l.84-1.102h-.84v-1.02h2.21v1.896l-.832 1.096h.832v1.001h-2.21zm2.385 0v-3.992h1.203v.836h.468c.556 0 1.011-.209 1.011-.836v-.836h.647l.001.802c.01.969-.68 1.469-1.347 1.469h-.492v2.557H6.601zm3.134 0h-1.203v-3.992h1.203v3.992zm1.677-3.992h2.092v.966h-1.227v.648h1.168v.936h-1.168v.767h1.227v.675h-2.092v-3.992zm3.582 3.992h-1.203v-3.992h1.203v3.992zm2.119 0h-1.084l-.832-1.639v1.639h-1.139v-3.992h1.1l.807 1.634v-1.634h1.148v3.992z" />
    </svg>
  );
}

export default function SocialButtons() {
  return (
    <div className="space-y-2">
      <button
        onClick={() => signIn("google", { callbackUrl: "/" })}
        className="w-full py-2 bg-[#1E1B18] hover:bg-[#2C2824] border border-[#36322E] text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
      >
        <GoogleIcon />
        Continue with Google
      </button>
      <button
        onClick={() => signIn("facebook", { callbackUrl: "/" })}
        className="w-full py-2 bg-[#1E1B18] hover:bg-[#2C2824] border border-[#36322E] text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
      >
        <FacebookIcon />
        Continue with Facebook
      </button>
      <button
        onClick={() => signIn("line", { callbackUrl: "/" })}
        className="w-full py-2 bg-[#1E1B18] hover:bg-[#2C2824] border border-[#36322E] text-white text-xs font-semibold rounded-lg flex items-center justify-center gap-2 transition cursor-pointer"
      >
        <LineIcon />
        Continue with LINE
      </button>
    </div>
  );
}
