import type { Metadata } from "next";
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
      <body className="bg-[#0A0A0A] text-[#F3F3F3] antialiased">
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
