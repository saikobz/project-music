import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

// ลงทะเบียนฟอนต์หลักไว้ที่ root layout เพื่อให้ทั้งแอปใช้ตัวแปรฟอนต์ชุดเดียวกันได้
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HarmoniQ",
  description: "Web Application for Music Source Separation Using AI with Automatic EQ and Compressor",
};

// layout หลักของแอป ทำหน้าที่ห่อทุกหน้าและใส่สไตล์ระดับทั้งระบบ
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
