import type { Metadata } from "next";
import { Toaster } from "sonner";
import { Inter, Space_Grotesk, Geist_Mono } from "next/font/google";
import AuthProvider from "./components/SessionProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "HarmoniQ — AI Audio Toolkit",
  description: "ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI",
};

// ฟอนต์ตาม DESIGN.md — Display: Space Grotesk, Body: Inter, Mono: Geist Mono
// หมายเหตุ: ฟอนต์ทั้ง 3 ไม่มีชุดอักษรไทย — ตัวอักษรไทยจะ fallback ไปใช้ระบบอัตโนมัติ
const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  display: "swap",
});

const bodyFont = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const monoFont = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-geist-mono",
  display: "swap",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="th"
      className={`dark ${displayFont.variable} ${bodyFont.variable} ${monoFont.variable}`}
    >
      <body className="bg-[#0D0B0A] text-[#F5F0EB] antialiased">
        <AuthProvider>{children}</AuthProvider>
        <Toaster position="bottom-right" richColors />
      </body>
    </html>
  );
}

