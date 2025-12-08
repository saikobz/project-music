"use client";
import React, { useRef, useState } from "react";
import axios from "axios";
import WaveformPlayer from "./WaveformPlayer";
import MultiStemLivePlayer from "./MultiStemLivePlayer";
import AudioAnalysis from "./AudioAnalysis";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const MAX_SIZE_BYTES = 100 * 1024 * 1024; // 100MB

function UploadBox() {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [action, setAction] = useState("separate");
  const [target, setTarget] = useState("vocals");
  const [strength, setStrength] = useState("medium");
  const [genre, setGenre] = useState("general");
  const [pitchSteps, setPitchSteps] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<{ tempo: number; key: string; pitch: string | null } | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const progressTimerRef = useRef<NodeJS.Timeout | null>(null);

  const handleFileSelect = (selected: File | null) => {
    setErrorMessage(null);
    setSuccessMessage(null);
    setFile(null);
    if (!selected) return;
    const ext = selected.name.toLowerCase().split(".").pop();
    if (ext !== "wav") {
      setErrorMessage("รองรับเฉพาะไฟล์ WAV (.wav)");
      return;
    }
    if (selected.size > MAX_SIZE_BYTES) {
      setErrorMessage("ไฟล์ต้องมีขนาดไม่เกิน 100MB");
      return;
    }
    setFile(selected);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files?.[0];
    handleFileSelect(droppedFile || null);
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setErrorMessage("กรุณาเลือกไฟล์ WAV (≤100MB) ก่อนเริ่มประมวลผล");
      return;
    }

    setLoading(true);
    setDownloadUrl(null);
    setDownloadFileName(null);
    setProcessingTime(null);
    setFileId(null);
    setZipUrl(null);
    setErrorMessage(null);
    setSuccessMessage(null);
    setAnalysis(null);
    setStatusText("กำลังอัปโหลดและประมวลผล...");
    setProgress(0);
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
    }
    // Simulate progress bar while waiting for backend response
    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return prev; // stop at 90% until completion
        return prev + 2;
      });
    }, 200);

    const formData = new FormData();
    formData.append("file", file);

    const startTime = Date.now();

    try {
      let response;
      let suffix = "";

      if (action === "separate") {
        response = await axios.post(`${API_BASE}/separate`, formData);
        const { file_id, zip_url } = response.data;
        setFileId(file_id);
        setZipUrl(zip_url);
        setSuccessMessage("แยกสเตมเสร็จแล้ว ดาวน์โหลด ZIP หรือลองเล่นทีละสเตมได้เลย");
        setStatusText("กำลังเตรียมไฟล์สเตม...");
      }

      if (action === "eq") {
        response = await axios.post(`${API_BASE}/apply-eq?target=${target}&genre=${genre}`, formData, {
          responseType: "blob",
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        suffix = `_eq_${target}`;
        setSuccessMessage("ประมวลผล EQ เสร็จแล้ว ดาวน์โหลดไฟล์เพื่อฟังผลลัพธ์ได้");
        setStatusText("กำลังสร้างไฟล์ EQ...");
      }

      if (action === "compressor") {
        response = await axios.post(`${API_BASE}/apply-compressor?strength=${strength}&genre=${genre}`, formData, {
          responseType: "blob",
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        suffix = `_compressed_${strength}`;
        setSuccessMessage("ประมวลผล Compressor เสร็จแล้ว");
        setStatusText("กำลังสร้างไฟล์ Compressor...");
      }

      if (action === "pitch") {
        response = await axios.post(`${API_BASE}/pitch-shift?steps=${pitchSteps}`, formData, {
          responseType: "blob",
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        suffix = `_pitch_${pitchSteps}`;
        setSuccessMessage("ประมวลผล Pitch Shift เสร็จแล้ว");
        setStatusText("กำลังสร้างไฟล์ Pitch Shift...");
      }

      if (file && suffix) {
        const baseName = file.name.replace(/\.[^/.]+$/, "");
        setDownloadFileName(`${baseName}${suffix}.wav`);
      }

      const analyzeData = new FormData();
      analyzeData.append("file", file);
      try {
        const analyzeResp = await axios.post(`${API_BASE}/analyze`, analyzeData);
        setAnalysis(analyzeResp.data);
      } catch (err) {
        console.error("Analyze error", err);
      }

      const endTime = Date.now();
      const duration = Math.floor((endTime - startTime) / 1000);
      const minutes = Math.floor(duration / 60);
      const seconds = duration % 60;
      setProcessingTime(`${minutes} นาที ${seconds} วินาที`);
      setProgress(100);
      setStatusText("เสร็จแล้ว! ดาวน์โหลดหรือเล่นไฟล์ได้เลย");
      setSuccessMessage((prev) => prev || "ประมวลผลเสร็จแล้ว");
    } catch (err: any) {
      let message = "เกิดข้อผิดพลาดระหว่างประมวลผล กรุณาลองใหม่";
      if (err?.response?.data?.detail) {
        message = err.response.data.detail;
      } else if (err.code === "ERR_NETWORK") {
        message = "ติดต่อ backend ไม่ได้ (ตรวจสอบการรันเซิร์ฟเวอร์หรือ CORS)";
      } else if (err.response?.status) {
        const status = err.response.status;
        if (status >= 500) {
          message = `เซิร์ฟเวอร์มีปัญหา (${status})`;
        } else {
          message = `คำขอไม่สำเร็จ (${status})`;
        }
      }
      setErrorMessage(message);
      setStatusText(null);
      setProgress(100);
    } finally {
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
        progressTimerRef.current = null;
      }
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 md:grid-cols-2 p-6 text-[#EDE9FE]">
      <div className="space-y-4">
        <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur shadow-inner shadow-purple-900/30">
          <p className="text-sm text-[#A78BFA]">ขั้นตอนที่ 1</p>
          <h2 className="text-2xl font-bold">เลือกไฟล์ WAV</h2>
          <p className="text-sm text-[#EDE9FE]/80">รองรับไฟล์ .wav ขนาดไม่เกิน 100MB</p>

          <label
            className={`mt-3 flex h-28 cursor-pointer items-center justify-center rounded-xl border-2 border-dashed text-center text-[#EDE9FE] transition
            ${isDragging ? "border-[#22D3EE] bg-[#5B21B6]/40" : "border-[#8B5CF6] bg-[#5B21B6]/20 hover:border-[#22D3EE] hover:bg-[#5B21B6]/30"}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              type="file"
              accept="audio/wav"
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
            />
            {file ? (
              <span className="font-semibold">{file.name}</span>
            ) : (
              <span>ลากไฟล์มาวางหรือคลิกเพื่อเลือกไฟล์</span>
            )}
          </label>
        </div>

        <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur space-y-3 shadow-inner shadow-purple-900/30">
          <p className="text-sm text-[#A78BFA]">ขั้นตอนที่ 2</p>
          <h3 className="text-xl font-semibold">เลือกงานที่ต้องการ</h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { value: "separate", label: "แยกสเตม" },
              { value: "eq", label: "EQ" },
              { value: "compressor", label: "Compressor" },
              { value: "pitch", label: "ปรับ Pitch" },
            ].map((item) => (
              <button
                key={item.value}
                onClick={() => setAction(item.value)}
                className={`rounded-lg px-3 py-2 text-sm font-semibold border cursor-pointer transition ${
                  action === item.value
                    ? "bg-[#5B21B6] text-white border-[#22D3EE]"
                    : "bg-[#111827] border-[#5B21B6]/50 text-[#EDE9FE]"
                }`}
                disabled={loading}
              >
                {item.label}
              </button>
            ))}
          </div>

          {action === "eq" && (
            <div>
              <label className="block text-sm mb-1">Preset EQ</label>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                disabled={loading}
              >
                <option value="vocals">Vocals</option>
                <option value="drums">Drums</option>
                <option value="bass">Bass</option>
                <option value="other">Other</option>
              </select>
            </div>
          )}

          {action === "compressor" && (
            <div>
              <label className="block text-sm mb-1">ความแรง (Strength)</label>
              <select
                value={strength}
                onChange={(e) => setStrength(e.target.value)}
                className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                disabled={loading}
              >
                <option value="soft">Soft</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
          )}

          {(action === "eq" || action === "compressor") && (
            <div>
              <label className="block text-sm mb-1">แนวเพลง (Genre)</label>
              <select
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                disabled={loading}
              >
                <option value="general">ทั่วไป</option>
                <option value="pop">Pop</option>
                <option value="rock">Rock</option>
                <option value="trap">Trap</option>
                <option value="country">Country</option>
                <option value="soul">Soul</option>
              </select>
            </div>
          )}

          {action === "pitch" && (
            <div>
              <label className="block text-sm mb-1">ปรับ pitch (half-steps ±)</label>
              <input
                type="number"
                value={pitchSteps}
                onChange={(e) => setPitchSteps(parseFloat(e.target.value))}
                className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                disabled={loading}
              />
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={loading}
            className={`w-full rounded-xl py-3 text-lg font-bold transition ${
              loading ? "bg-[#A78BFA]/60 cursor-not-allowed" : "bg-[#5B21B6] hover:bg-[#22D3EE] text-white cursor-pointer"
            }`}
          >
            {loading ? "กำลังประมวลผล..." : "เริ่มประมวลผล"}
          </button>
          <div className="h-2 w-full rounded-full bg-[#0B1021] overflow-hidden border border-[#5B21B6]/40">
            <div
              className="h-full bg-gradient-to-r from-[#5B21B6] via-[#22D3EE] to-[#5B21B6] transition-[width] duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-[#A78BFA]">
            <span>สถานะ: {statusText || (loading ? "กำลังประมวลผล..." : "รอเริ่มงาน")}</span>
            <span>{progress}%</span>
          </div>
          {statusText && (
            <div className="rounded-lg bg-[#0F0B1D]/70 border border-[#7C3AED]/30 p-2 text-sm text-[#A78BFA]">
              {statusText}
            </div>
          )}
          {processingTime && (
            <div className="text-sm text-[#A78BFA]">ใช้เวลา: {processingTime}</div>
          )}
          {errorMessage && (
            <div className="rounded-lg bg-red-700/70 border border-red-400 p-2 text-sm text-white">
              {errorMessage}
            </div>
          )}
          {successMessage && (
            <div className="rounded-lg bg-[#7C3AED]/30 border border-[#A78BFA] p-2 text-sm text-[#EDE9FE]">
              {successMessage}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {analysis && <AudioAnalysis data={analysis} />}

        {fileId && (
          <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur">
            <h3 className="text-xl font-semibold mb-2">Multi-stem Player</h3>
            <MultiStemLivePlayer fileId={fileId} />
          </div>
        )}

        {zipUrl && (
          <a
            href={zipUrl}
            download="separated.zip"
            className="block w-full text-center rounded-xl bg-[#22D3EE] hover:bg-[#5B21B6] text-black font-semibold py-3"
          >
            ดาวน์โหลดสเตมทั้งหมด (ZIP)
          </a>
        )}

        {downloadUrl && downloadFileName && !downloadFileName.endsWith(".zip") && (
          <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur space-y-3">
            <a
              href={downloadUrl}
              download={downloadFileName}
              className="block w-full text-center rounded-xl bg-[#5B21B6] hover:bg-[#22D3EE] text-white font-semibold py-3"
            >
              ดาวน์โหลดไฟล์ (WAV)
            </a>
            <WaveformPlayer audioUrl={downloadUrl} />
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadBox;
