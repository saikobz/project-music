import type { Metadata } from "next";
import { Toaster } from "sonner";
import AuthProvider from "./components/SessionProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "HarmoniQ — AI Audio Toolkit",
  description: "ระบบแยกแทร็กเสียงดนตรีและปรับแต่งเสียงด้วย AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="th" className="dark">
      <body className="bg-[#0D0B0A] text-[#F5F0EB] antialiased">
        <AuthProvider>{children}</AuthProvider>
        <Toaster position="bottom-right" richColors />
      </body>
    </html>
  );
}

