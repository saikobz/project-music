// หน้าที่ของไฟล์นี้:
// - เป็นฟอร์มหลักของหน้าเว็บสำหรับรับไฟล์ WAV และเลือกประเภทการประมวลผล
// - เรียก backend ตาม action ที่ผู้ใช้เลือก เช่น แยก stem, Auto-EQ, Compressor และ Pitch Shift
// - จัดการสถานะของ UI เช่น progress, error, ผลวิเคราะห์, ลิงก์ดาวน์โหลด และตัวเล่นผลลัพธ์
"use client";
import React, { useRef, useState } from "react";
import axios from "axios";
import WaveformPlayer from "./WaveformPlayer";
import MultiStemLivePlayer from "./MultiStemLivePlayer";
import AudioAnalysis from "./AudioAnalysis";
import { AutoEqSettings } from "./settings/AutoEqSettings";
import { CompressorSettings } from "./settings/CompressorSettings";
import { PitchShiftSettings } from "./settings/PitchShiftSettings";
// ที่อยู่ของ backend และข้อจำกัดขนาดไฟล์ฝั่งหน้าเว็บ

// ค่าตั้งต้นของ API และข้อจำกัดการอัปโหลด
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const MAX_SIZE_BYTES = 100 * 1024 * 1024; // 100MB
const AUTO_EQ_DELTA_CLAMP_MIN = 0;
const AUTO_EQ_DELTA_CLAMP_MAX = 6;
const AUTO_EQ_DELTA_CLAMP_DEFAULT = 2;
const AUTO_EQ_MODEL_DEFAULT = "lstm-last";
const AUTO_EQ_MODEL_OPTIONS = [
  { value: "cnn-v1", label: "CNN", hint: "โหมดเดิมของโปรเจกต์" },
  { value: "lstm-last", label: "LSTM", hint: "โมเดลใหม่แบบ sequence-aware" },
];

// ===== Skeleton Components สำหรับแสดงสถานะกำลังประมวลผล (Processing State) =====
function AudioAnalysisSkeleton() {
  return (
    <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 shadow-lg animate-pulse">
      <div className="flex items-center gap-2 mb-4">
        <div className="h-5 w-28 bg-[#2A2A2A] rounded"></div>
        <div className="h-5 w-24 bg-[#2A2A2A] rounded opacity-50"></div>
      </div>
      <div className="grid grid-cols-3 gap-4 text-center">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-lg bg-[#1A1A1A] p-3 border border-[#2A2A2A] space-y-2 flex flex-col items-center">
            <div className="h-3 w-10 bg-[#2A2A2A] rounded"></div>
            <div className="h-6 w-16 bg-[#2A2A2A] rounded"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StemMixerSkeleton() {
  return (
    <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 shadow-lg animate-pulse space-y-4">
      <div className="flex items-center justify-between">
        <div className="h-5 w-32 bg-[#2A2A2A] rounded"></div>
        <div className="flex gap-2">
          <div className="h-8 w-24 bg-[#2A2A2A] rounded-lg"></div>
          <div className="h-8 w-20 bg-[#2A2A2A] rounded-lg"></div>
        </div>
      </div>
      <div className="space-y-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="rounded-xl border border-[#2A2A2A] bg-[#1A1A1A] p-3 space-y-3">
            <div className="flex justify-between items-center">
              <div className="space-y-1.5">
                <div className="h-4 w-16 bg-[#2A2A2A] rounded"></div>
                <div className="h-3 w-28 bg-[#2A2A2A] rounded opacity-60"></div>
              </div>
              <div className="flex gap-2">
                <div className="h-6 w-16 bg-[#2A2A2A] rounded-lg"></div>
                <div className="h-6 w-12 bg-[#2A2A2A] rounded-lg"></div>
              </div>
            </div>
            <div className="h-14 w-full bg-[#121212] rounded border border-[#2A2A2A] opacity-80 flex items-center px-3">
              <div className="h-2 w-12 bg-[#2A2A2A] rounded mr-3"></div>
              <div className="h-1.5 flex-grow bg-[#2A2A2A] rounded"></div>
              <div className="h-4 w-10 bg-[#2A2A2A] rounded ml-3"></div>
            </div>
            <div className="h-10 w-full bg-[#121212] rounded border border-[#2A2A2A] opacity-40"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SinglePlayerSkeleton() {
  return (
    <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 shadow-lg animate-pulse space-y-4">
      <div className="h-12 w-full bg-[#E5A93D]/10 rounded-lg border border-[#E5A93D]/20 flex items-center justify-center">
        <div className="h-4 w-40 bg-[#E5A93D]/30 rounded"></div>
      </div>
      <div className="h-24 w-full bg-[#1A1A1A] rounded-lg border border-[#2A2A2A] opacity-60"></div>
    </div>
  );
}

interface UploadBoxProps {
  onHeightChange?: (expanded: boolean) => void;
}

function UploadBox({ onHeightChange }: UploadBoxProps) {
  // ===== กลุ่ม state สำหรับค่าที่ผู้ใช้เลือกในฟอร์ม =====
  // สถานะกลุ่มนี้กำหนดว่าจะเรียก backend action ไหนและใช้พารามิเตอร์อะไร
  // สถานะของไฟล์และตัวเลือกการประมวลผล
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [action, setAction] = useState("separate");
  const [strength, setStrength] = useState("medium");
  const [genre, setGenre] = useState("pop");
  const [autoEqModel, setAutoEqModel] = useState(AUTO_EQ_MODEL_DEFAULT);
  const [deltaClampDb, setDeltaClampDb] = useState(String(AUTO_EQ_DELTA_CLAMP_DEFAULT));
  const [compThreshold, setCompThreshold] = useState("");
  const [compRatio, setCompRatio] = useState("");
  const [compAttack, setCompAttack] = useState("");
  const [compRelease, setCompRelease] = useState("");
  const [compKnee, setCompKnee] = useState("6");
  const [compMakeupGain, setCompMakeupGain] = useState("0");
  const [compDryWet, setCompDryWet] = useState("100");
  const [compOutputCeiling, setCompOutputCeiling] = useState("");
  const [pitchSteps, setPitchSteps] = useState(0);

  // สถานะผลลัพธ์จาก backend ใช้กับการดาวน์โหลด การเล่นไฟล์ และเวลาที่ใช้ประมวลผล
  // สถานะผลลัพธ์สำหรับดาวน์โหลดและผลวิเคราะห์
  // ===== กลุ่ม state สำหรับผลลัพธ์ที่ backend ส่งกลับมา =====
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

  // สถานะของ UI ล้วน ๆ เช่น banner ข้อความ, ผลวิเคราะห์, และ progress จำลอง
  // สถานะข้อความและการตอบสนองของ UI
  // ===== กลุ่ม state สำหรับสถานะของหน้า =====
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<{ tempo: number; key: string; pitch: string | null } | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const progressTimerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  React.useEffect(() => {
    if (onHeightChange) {
      const expanded = action !== "separate" || !!file || loading || !!zipUrl || !!downloadUrl || !!analysis;
      onHeightChange(expanded);
    }
  }, [action, file, loading, zipUrl, downloadUrl, analysis, onHeightChange]);

  React.useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (loading) {
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
        e.preventDefault();
        e.returnValue = "";
      }
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [loading]);

  React.useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
    };
  }, []);

  // ตัวตรวจสอบกลาง ใช้ร่วมกันทั้ง input file และ drag-and-drop
  // ตัวช่วยสำหรับเลือกไฟล์และลากวาง
  // ===== ตัวช่วยรับไฟล์จาก input และ drag-and-drop =====
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

  const eqDeltaClampValue = Number.parseFloat(deltaClampDb);
  const isEqDeltaClampValid =
    Number.isFinite(eqDeltaClampValue) &&
    eqDeltaClampValue >= AUTO_EQ_DELTA_CLAMP_MIN &&
    eqDeltaClampValue <= AUTO_EQ_DELTA_CLAMP_MAX;
  const eqDeltaClampWarning = !isEqDeltaClampValid
    ? `กรอกค่า Delta Clamp ระหว่าง ${AUTO_EQ_DELTA_CLAMP_MIN.toFixed(1)} ถึง ${AUTO_EQ_DELTA_CLAMP_MAX.toFixed(1)} dB`
    : eqDeltaClampValue > 3.5
      ? "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ยิ่งปรับสูง Auto-EQ จะยิ่งแก้หนักขึ้น เสี่ยงโทนแข็ง บวม หรือเสียงแตกได้ง่ายขึ้น"
      : eqDeltaClampValue < 1.0
        ? "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ค่าต่ำจะคุมการปรับไว้เบาและปลอดภัยกว่า แต่ความเปลี่ยนแปลงของเสียงอาจไม่ชัดมาก"
        : "Delta Clamp คือเพดานการบูสต์/คัต EQ สูงสุดต่อย่านความถี่ ค่าช่วงกลางมักบาลานซ์ระหว่างความชัดของผลลัพธ์กับความเป็นธรรมชาติของเสียง";

  const selectedAutoEqModel =
    AUTO_EQ_MODEL_OPTIONS.find((option) => option.value === autoEqModel) ?? AUTO_EQ_MODEL_OPTIONS[0];

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  // โฟลว์หลักของการส่งงาน:
  // 1. ล้างผลลัพธ์เก่า
  // 2. เรียก endpoint ตาม action ที่เลือก
  // 3. เก็บ URL หรือ id ที่ได้กลับมาเพื่อใช้แสดงผลด้านขวา
  // 4. วิเคราะห์ไฟล์ต้นฉบับเพื่อหา tempo, key และ pitch
  // โฟลว์หลักสำหรับประมวลผลทุก action
  // ===== ฟังก์ชันหลัก: ส่งไฟล์ไป backend และอัปเดตผลลัพธ์บนหน้า =====
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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    // ทำ progress แบบจำลองไว้ก่อน เพราะ backend ไม่ได้ส่งสถานะระหว่างประมวลผลกลับมา
    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) return prev; // หยุดไว้ที่ 90% แล้วรอผลจริงจาก backend
        return prev + 2;
      });
    }, 200);

    const formData = new FormData();
    formData.append("file", file);

    const startTime = Date.now();

    try {
      let response;
      let suffix = "";

      // งานแยก stem จะได้ทั้ง file id และ URL สำหรับดาวน์โหลด ZIP
      // แตก flow ตาม action เพื่อให้ UI ตัวเดียวรองรับหลาย endpoint
      if (action === "separate") {
        response = await axios.post(`${API_BASE}/separate`, formData, { signal });
        const { file_id, zip_url } = response.data;
        setFileId(file_id);
        setZipUrl(zip_url);
        setSuccessMessage("แยกเสียงเสร็จแล้ว ดาวน์โหลด ZIP หรือลองเล่นทีละสเตมได้เลย");
        setStatusText("กำลังเตรียมไฟล์สเตม...");
      }

      // action อื่น ๆ จะคืนไฟล์ WAV เดี่ยวกลับมาในรูป blob สำหรับสร้างลิงก์ดาวน์โหลด
      if (action === "eq-ai") {
        const params = new URLSearchParams({
          genre,
          model_id: autoEqModel,
          delta_clamp_db: deltaClampDb || String(AUTO_EQ_DELTA_CLAMP_DEFAULT),
        });
        response = await axios.post(`${API_BASE}/apply-eq-ai?${params.toString()}`, formData, {
          responseType: "blob",
          signal,
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        suffix = `_eq_ai_${autoEqModel}_${genre}_${deltaClampDb}db`;
        setSuccessMessage(`Auto-EQ (${selectedAutoEqModel.label}) completed successfully.`);
        setStatusText(`Running Auto-EQ with ${selectedAutoEqModel.label}...`);
      }

      if (action === "compressor") {
        const params = new URLSearchParams({
          strength,
          genre,
          knee: compKnee || "6",
          makeup_gain: compMakeupGain || "0",
          dry_wet: compDryWet || "100",
        });
        if (compThreshold) params.set("threshold", compThreshold);
        if (compRatio) params.set("ratio", compRatio);
        if (compAttack) params.set("attack", compAttack);
        if (compRelease) params.set("release", compRelease);
        if (compOutputCeiling) params.set("output_ceiling", compOutputCeiling);

        response = await axios.post(`${API_BASE}/apply-compressor?${params.toString()}`, formData, {
          responseType: "blob",
          signal,
        });
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        suffix = `_compressed_${strength}`;
        setSuccessMessage("ประมวลผล Compressor เสร็จแล้ว");
        setStatusText("กำลังสร้างไฟล์ Compressor...");
      }

      if (action === "pitch") {
        response = await axios.post(`${API_BASE}/pitch-shift?steps=${pitchSteps}`, formData, {
          responseType: "blob", //"blob" หมายถึง บอก axios ว่า response ที่ backend ส่งกลับมาเป็น “ข้อมูลไฟล์ดิบ” ไม่ใช่ JSON หรือ text
          signal,
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

      // การวิเคราะห์ถูกแยกรันอีกครั้ง เพื่อให้การ์ดสรุปแสดงข้อมูลของไฟล์ต้นฉบับได้
      // ไม่ว่าผู้ใช้จะเลือก action ไหนก็ตาม
      // วิเคราะห์ tempo / key / pitch หลังงานหลักเสร็จ โดยใช้ไฟล์ต้นฉบับก้อนเดียวกัน
      const analyzeData = new FormData();
      analyzeData.append("file", file);
      try {
        const analyzeResp = await axios.post(`${API_BASE}/analyze`, analyzeData, { signal });
        setAnalysis(analyzeResp.data);
      } catch (err) {
        if (axios.isCancel(err)) {
          console.log("Analyze request canceled");
        } else {
          console.error("Analyze error", err);
        }
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
      if (axios.isCancel(err)) {
        console.log("Request canceled by user or refresh");
        return;
      }
      // รวมรูปแบบ error หลายแบบให้เหลือข้อความที่ผู้ใช้เข้าใจได้ง่ายบนหน้าเว็บ
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
    <div className={`p-6 text-[#F3F3F3] transition-all duration-500 ${!file ? "flex flex-col items-center justify-center min-h-[50vh]" : "grid gap-6 lg:grid-cols-12"}`}>
      {!file ? (
        <div className="w-full max-w-3xl text-center space-y-6">
          <h2 className="text-3xl font-bold tracking-tight text-[#F3F3F3]">Upload Audio</h2>
          <p className="text-[#8E8E8E] font-medium">Select or drag & drop a WAV file (up to 100MB)</p>
          <label
            className={`mt-6 mx-auto flex h-64 max-w-2xl cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed transition-all
            ${isDragging ? "border-[#E5A93D] bg-[#E5A93D]/10" : "border-[#2A2A2A] bg-[#121212] hover:border-[#E5A93D]/50 hover:bg-[#1A1A1A]"}`}
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
            <div className="text-lg font-medium text-[#F3F3F3]">Click to browse or drag file here</div>
            <div className="mt-2 text-sm text-[#555555]">.wav only</div>
          </label>
          {errorMessage && (
            <div className="rounded-lg bg-red-900/30 border border-red-900/50 p-3 text-sm text-red-400 max-w-xl mx-auto">
              {errorMessage}
            </div>
          )}
        </div>
      ) : (
        <>
          <div className="space-y-4 lg:col-span-4">
            <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-4 shadow-lg">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs font-medium text-[#8E8E8E] uppercase tracking-wider mb-1">Source File</p>
                  <h2 className="text-lg font-semibold text-[#F3F3F3] truncate max-w-[200px] sm:max-w-[300px]">{file.name}</h2>
                </div>
                <button 
                  onClick={() => handleFileSelect(null)}
                  disabled={loading}
                  className={`text-xs px-3 py-1.5 rounded bg-[#1A1A1A] border border-[#2A2A2A] text-[#8E8E8E] transition ${
                    loading 
                      ? "opacity-50 cursor-not-allowed" 
                      : "hover:text-[#F3F3F3] hover:border-[#8E8E8E] cursor-pointer"
                  }`}
                >
                  Change File
                </button>
              </div>
            </div>

            <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-4 space-y-4 shadow-lg">
              <div>
                <p className="text-xs font-medium text-[#8E8E8E] uppercase tracking-wider mb-2">Processing Module</p>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { value: "separate", label: "Stem Separation" },
                    { value: "eq-ai", label: "Auto EQ (AI)" },
                    { value: "compressor", label: "Compressor" },
                    { value: "pitch", label: "Pitch Shift" },
                  ].map((item) => (
                    <button
                      key={item.value}
                      onClick={() => setAction(item.value)}
                      className={`rounded-lg px-3 py-2.5 text-sm font-medium border cursor-pointer transition ${
                        action === item.value
                          ? "bg-[#E5A93D] text-[#0A0A0A] border-[#E5A93D]"
                          : "bg-[#1A1A1A] border-[#2A2A2A] text-[#8E8E8E] hover:text-[#F3F3F3] hover:border-[#555555]"
                      }`}
                      disabled={loading}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              {(action === "eq-ai" || action === "compressor") && (
                <div>
                  <label className="block text-xs font-medium text-[#8E8E8E] uppercase tracking-wider mb-2">Genre Profile</label>
                  <select
                    value={genre}
                    onChange={(e) => setGenre(e.target.value)}
                    className="w-full rounded-lg bg-[#0A0A0A] border border-[#2A2A2A] p-2.5 text-[#F3F3F3] focus:border-[#E5A93D] focus:outline-none transition"
                    disabled={loading}
                  >
                    <option value="pop">Pop</option>
                    <option value="rock">Rock</option>
                    <option value="trap">Trap</option>
                    <option value="country">Country</option>
                    <option value="soul">Soul</option>
                  </select>
                </div>
              )}

              {action === "eq-ai" && (
                <AutoEqSettings
                  autoEqModel={autoEqModel}
                  setAutoEqModel={setAutoEqModel}
                  deltaClampDb={deltaClampDb}
                  setDeltaClampDb={setDeltaClampDb}
                  loading={loading}
                  modelOptions={AUTO_EQ_MODEL_OPTIONS}
                  minDeltaClamp={AUTO_EQ_DELTA_CLAMP_MIN}
                  maxDeltaClamp={AUTO_EQ_DELTA_CLAMP_MAX}
                  defaultDeltaClamp={AUTO_EQ_DELTA_CLAMP_DEFAULT}
                  isValid={isEqDeltaClampValid}
                  warningText={eqDeltaClampWarning}
                />
              )}

              {action === "compressor" && (
                <CompressorSettings
                  strength={strength}
                  setStrength={setStrength}
                  compThreshold={compThreshold}
                  setCompThreshold={setCompThreshold}
                  compRatio={compRatio}
                  setCompRatio={setCompRatio}
                  compAttack={compAttack}
                  setCompAttack={setCompAttack}
                  compRelease={compRelease}
                  setCompRelease={setCompRelease}
                  compKnee={compKnee}
                  setCompKnee={setCompKnee}
                  compMakeupGain={compMakeupGain}
                  setCompMakeupGain={setCompMakeupGain}
                  compDryWet={compDryWet}
                  setCompDryWet={setCompDryWet}
                  compOutputCeiling={compOutputCeiling}
                  setCompOutputCeiling={setCompOutputCeiling}
                  loading={loading}
                />
              )}

              {action === "pitch" && (
                <PitchShiftSettings
                  pitchSteps={pitchSteps}
                  setPitchSteps={setPitchSteps}
                  loading={loading}
                />
              )}

              <button
                onClick={handleUpload}
                disabled={loading || (action === "eq-ai" && !isEqDeltaClampValid)}
                className={`w-full rounded-lg py-3 text-sm font-bold uppercase tracking-wider transition ${
                  loading || (action === "eq-ai" && !isEqDeltaClampValid)
                    ? "bg-[#2A2A2A] text-[#555555] cursor-not-allowed border border-[#2A2A2A]"
                    : "bg-[#E5A93D] hover:bg-[#F3C05D] text-[#0A0A0A] cursor-pointer shadow-[0_0_15px_rgba(229,169,61,0.2)] hover:shadow-[0_0_20px_rgba(229,169,61,0.4)]"
                }`}
              >
                {loading ? "Processing..." : "Execute"}
              </button>

              <div className="h-1.5 w-full rounded-full bg-[#0A0A0A] overflow-hidden border border-[#2A2A2A]">
                <div
                  className="h-full bg-[#E5A93D] transition-[width] duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-[#8E8E8E]">
                <span>{statusText || (loading ? "Processing..." : "Ready")}</span>
                <span>{progress}%</span>
              </div>

              {statusText && (
                <div className="rounded-lg bg-[#1A1A1A] border border-[#2A2A2A] p-2.5 text-xs text-[#F3F3F3]">
                  {statusText}
                </div>
              )}
              {processingTime && (
                <div className="text-xs text-[#8E8E8E]">Processing Time: {processingTime}</div>
              )}
              {errorMessage && (
                <div className="rounded-lg bg-red-900/30 border border-red-900/50 p-2.5 text-xs text-red-400">
                  {errorMessage}
                </div>
              )}
              {successMessage && (
                <div className="rounded-lg bg-[#E5A93D]/10 border border-[#E5A93D]/30 p-2.5 text-xs text-[#E5A93D]">
                  {successMessage}
                </div>
              )}
            </div>
          </div>

          <div className="space-y-4 lg:col-span-8">
            {loading && (
              <>
                <AudioAnalysisSkeleton />
                {action === "separate" ? <StemMixerSkeleton /> : <SinglePlayerSkeleton />}
              </>
            )}

            {!loading && analysis && <AudioAnalysis data={analysis} />}

            {!loading && fileId && (
              <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 shadow-lg">
                <h3 className="text-sm font-medium text-[#8E8E8E] uppercase tracking-wider mb-4">Stem Mixer</h3>
                <MultiStemLivePlayer fileId={fileId} />
              </div>
            )}

            {!loading && zipUrl && (
              <a
                href={zipUrl.startsWith("http") ? zipUrl : `${API_BASE}${zipUrl}`}
                download="separated.zip"
                className="block w-full text-center rounded-lg bg-[#1A1A1A] hover:bg-[#2A2A2A] border border-[#2A2A2A] hover:border-[#555555] text-[#F3F3F3] font-medium py-3 transition"
              >
                Download All Stems (ZIP)
              </a>
            )}

            {!loading && downloadUrl && downloadFileName && !downloadFileName.endsWith(".zip") && (
              <div className="rounded-xl border border-[#2A2A2A] bg-[#121212] p-5 space-y-4 shadow-lg">
                <a
                  href={downloadUrl}
                  download={downloadFileName}
                  className="block w-full text-center rounded-lg bg-[#E5A93D] hover:bg-[#F3C05D] text-[#0A0A0A] font-medium py-3 transition shadow-[0_0_15px_rgba(229,169,61,0.2)]"
                >
                  Download Output (WAV)
                </a>
                <div className="pt-2">
                  <WaveformPlayer audioUrl={downloadUrl} />
                </div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default UploadBox;
