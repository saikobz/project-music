"use client";

import React, { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { Navbar } from "../../components/Navbar";
import { Footer } from "../../components/Footer";
import { Music, Download, Trash2, Clock, Disc, ArrowRight, Play, Pause } from "lucide-react";
import { toast } from "sonner";
import { API_BASE_URL } from "@/lib/config";
import { downloadViaBlob } from "@/lib/download";

interface HistoryRecord {
  id: string;
  action: string;
  originalFilename: string;
  fileId: string | null;
  stems: string | null;
  createdAt: string;
}

export default function HistoryPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [recordsLoading, setRecordsLoading] = useState(true);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  useEffect(() => () => audioRef.current?.pause(), []);

  useEffect(() => {
    fetch("/api/account")
      .then((res) => res.json())
      .then((d) => {
        setData(d);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!data || data.error) {
      setRecordsLoading(false);
      return;
    }
    fetch("/api/history")
      .then((res) => res.json())
      .then((d) => setRecords(d.records || []))
      .catch(() => setRecords([]))
      .finally(() => setRecordsLoading(false));
  }, [data]);

  const togglePlay = (fileId: string) => {
    if (playingId === fileId) {
      audioRef.current?.pause();
      setPlayingId(null);
    } else {
      if (audioRef.current) {
        audioRef.current.pause();
      }
      const url = `${API_BASE_URL}/separated/${fileId}/vocals.wav`;
      const audio = new Audio(url);
      audio.onended = () => setPlayingId(null);
      // M10: ไฟล์อาจถูกลบไปแล้วตาม TTL (~20 นาที) -> ต้องแจ้งผู้ใช้ ไม่ใช่เงียบ
      audio.onerror = () => {
        setPlayingId(null);
        toast.error("ไฟล์เสียงหมดอายุแล้ว (ระบบลบไฟล์อัตโนมัติ) กรุณาประมวลผลใหม่");
      };
      audio.play().catch(() => {
        setPlayingId(null);
        toast.error("ไม่สามารถเล่นไฟล์ได้ (ไฟล์อาจหมดอายุแล้ว) กรุณาประมวลผลใหม่");
      });
      audioRef.current = audio;
      setPlayingId(fileId);
    }
  };

  const handleDelete = async (id: string) => {
    const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
    if (res.ok) {
      setRecords((prev) => prev.filter((r) => r.id !== id));
    } else {
      toast.error("ไม่สามารถลบรายการได้ กรุณาลองใหม่");
    }
  };

  const handleDownload = async (fileId: string) => {
    // M9: ดาวน์โหลดผ่าน fetch -> blob (ลิงก์ข้าม origin + download attribute ไม่ทำงาน)
    const ok = await downloadViaBlob(`${API_BASE_URL}/download/${fileId}`, "separated.zip");
    if (!ok) {
      toast.error("ไฟล์หมดอายุแล้ว (ระบบลบไฟล์อัตโนมัติ) กรุณาประมวลผลใหม่");
    }
  };

  const actionLabels: Record<string, string> = {
    separate: "Stem Separation",
    "apply-eq-ai": "Auto-EQ",
    "apply-compressor": "Compressor",
    "pitch-shift": "Pitch Shift",
    analyze: "Audio Analysis",
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-4xl mx-auto py-24 text-center text-[#8E8E8E] text-sm">Loading history...</div>
        <Footer />
      </div>
    );
  }

  if (!data || data.error) {
    return (
      <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
        <Navbar />
        <div className="max-w-md mx-auto py-24 text-center space-y-4 px-4">
          <div className="w-12 h-12 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mx-auto text-purple-400">
            <Music className="w-6 h-6" />
          </div>
          <h2 className="text-xl font-bold">เข้าสู่ระบบเพื่อดูประวัติสเต็มเสียง</h2>
          <p className="text-xs text-[#8E8E8E]">
            กรุณาเข้าสู่ระบบเพื่อเข้าถึงรายการเพลงและแทร็กสเต็มที่คุณเคยแยกไว้
          </p>
          <Link
            href="/auth/signin"
            className="inline-flex items-center gap-2 px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white font-semibold text-xs rounded-xl transition"
          >
            <span>เข้าสู่ระบบ</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-[#F3F3F3] flex flex-col justify-between">
      <Navbar />

      <main className="flex-grow max-w-5xl mx-auto w-full px-4 py-12 space-y-8">
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#222] pb-6">
          <div>
            <div className="inline-flex items-center gap-2 text-purple-400 text-xs font-semibold uppercase tracking-wider mb-1">
              <Clock className="w-4 h-4" />
              <span>Cloud Stem Library</span>
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight">ประวัติการแยกแทร็กเสียง (Project History)</h1>
            <p className="text-[#8E8E8E] text-xs md:text-sm mt-1">
              รายการเพลงและไฟล์สเต็มที่ได้รับการประมวลผลไว้สำหรับดาวน์โหลด
            </p>
          </div>

          <Link
            href="/studio"
            className="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded-xl flex items-center gap-2 transition self-start md:self-auto"
          >
            <Disc className="w-4 h-4" />
            <span>+ แยกเพลงใหม่ใน Studio</span>
          </Link>
        </header>

        {recordsLoading ? (
          <div className="bg-[#111111] border border-[#222222] rounded-2xl p-12 text-center">
            <div className="text-[#8E8E8E] text-sm">Loading history...</div>
          </div>
        ) : records.length === 0 ? (
          <div className="bg-[#111111] border border-[#222222] rounded-2xl p-12 text-center space-y-4">
            <Music className="w-12 h-12 text-[#444] mx-auto" />
            <h3 className="text-lg font-semibold">ยังไม่มีประวัติการแยกแทร็กเสียง</h3>
            <p className="text-xs text-[#8E8E8E] max-w-sm mx-auto">
              อัปโหลดไฟล์ WAV ใน Studio Workspace เพื่อเริ่มแยกแทร็กเสียงดนตรีด้วย AI
            </p>
            <Link
              href="/studio"
              className="inline-block px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold rounded-xl transition"
            >
              ไปยัง Studio Workspace
            </Link>
          </div>
        ) : (
          <div className="bg-[#111111] border border-[#222222] rounded-2xl overflow-hidden shadow-xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs md:text-sm">
                <thead className="bg-[#181818] border-b border-[#222] text-[#888] font-medium uppercase text-xs">
                  <tr>
                    <th className="py-3.5 px-4">ชื่อไฟล์เพลง</th>
                    <th className="py-3.5 px-4">วันที่แยกแทร็ก</th>
                    <th className="py-3.5 px-4">รายละเอียด</th>
                    <th className="py-3.5 px-4 text-right">การจัดการ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1F1F1F]">
                  {records.map((record) => {
                    let stemsList: string[] = [];
                    try { stemsList = record.stems ? JSON.parse(record.stems) : []; } catch {}
                    const isSeparate = record.action === "separate";

                    return (
                      <tr key={record.id} className="hover:bg-[#161616] transition-colors">
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-3">
                            {isSeparate && record.fileId && (
                              <button
                                onClick={() => togglePlay(record.fileId!)}
                                className="w-9 h-9 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 flex items-center justify-center hover:bg-purple-500 hover:text-white transition"
                              >
                                {playingId === record.fileId ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                              </button>
                            )}
                            <div>
                              <p className="font-semibold text-white truncate max-w-xs">{record.originalFilename}</p>
                              <span className="text-[11px] text-purple-400">{actionLabels[record.action] || record.action}</span>
                            </div>
                          </div>
                        </td>
                        <td className="py-4 px-4 text-[#A0A0A0] text-xs">
                          {new Date(record.createdAt).toLocaleDateString("th-TH", {
                            year: "numeric", month: "short", day: "numeric",
                          })}
                        </td>
                        <td className="py-4 px-4">
                          {isSeparate && stemsList.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {stemsList.map((stem) => (
                                <span
                                  key={stem}
                                  className="px-2 py-0.5 rounded-md bg-[#202020] border border-[#303030] text-[11px] text-purple-300"
                                >
                                  {stem}
                                </span>
                              ))}
                            </div>
                          ) : (
                            <span className="text-[11px] text-[#555]">—</span>
                          )}
                        </td>
                        <td className="py-4 px-4 text-right">
                          <div className="flex items-center justify-end gap-2">
                            {isSeparate && record.fileId && (
                              <button
                                onClick={() => handleDownload(record.fileId!)}
                                className="p-2 rounded-lg bg-[#202020] hover:bg-[#303030] text-white transition text-xs flex items-center gap-1.5"
                              >
                                <Download className="w-4 h-4 text-purple-400" />
                                <span className="hidden sm:inline">ดาวน์โหลด</span>
                              </button>
                            )}
                            <button
                              onClick={() => handleDelete(record.id)}
                              className="p-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}
