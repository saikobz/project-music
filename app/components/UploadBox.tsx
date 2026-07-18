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
import ExportMasterModal from "./ExportMasterModal";
import SingleExportModal from "./SingleExportModal";
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

  const [isTrimming, setIsTrimming] = useState(false);
  const [trimStart, setTrimStart] = useState("0");
  const [trimEnd, setTrimEnd] = useState("30");
  const [exportFormat, setExportFormat] = useState("wav");

  // สถานะผลลัพธ์จาก backend ใช้กับการดาวน์โหลด การเล่นไฟล์ และเวลาที่ใช้ประมวลผล
  // สถานะผลลัพธ์สำหรับดาวน์โหลดและผลวิเคราะห์
  // ===== กลุ่ม state สำหรับผลลัพธ์ที่ backend ส่งกลับมา =====
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

  // สถานะสำหรับ Export Modal และ Mastering
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isSingleExportModalOpen, setIsSingleExportModalOpen] = useState(false);

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

  const handleExport = async (exportType: string, format: string, targetLufs: number, selectedStems: string[]) => {
    if (!fileId) return;
    setIsExporting(true);
    try {
      const queryParams = new URLSearchParams();
      queryParams.append("file_id", fileId);
      queryParams.append("export_type", exportType);
      queryParams.append("export_format", format);
      queryParams.append("target_lufs", targetLufs.toString());
      selectedStems.forEach((stem) => queryParams.append("stems", stem));

      const res = await fetch(`${API_BASE}/api/process/export?${queryParams.toString()}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.status === "success") {
        const url = data.file_url.startsWith("http") ? data.file_url : `${API_BASE}${data.file_url}`;
        const fileRes = await fetch(url);
        const blob = await fileRes.blob();
        const downloadUrlLocal = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = downloadUrlLocal;
        link.download = data.filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(downloadUrlLocal);
        setIsExportModalOpen(false);
      }
    } catch (err) {
      console.error("Export failed:", err);
    } finally {
      setIsExporting(false);
    }
  };

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
  const handleUpload = async (overrideFormat?: string | React.MouseEvent | React.FormEvent) => {
    if (!file) {
      setErrorMessage("กรุณาเลือกไฟล์ WAV (≤100MB) ก่อนเริ่มประมวลผล");
      return;
    }
    
    const actualFormat = typeof overrideFormat === "string" ? overrideFormat : exportFormat;

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
        const params = new URLSearchParams();
        if (isTrimming) {
          params.set("trim_start", trimStart);
          params.set("trim_end", trimEnd);
        }
        params.set("export_format", exportFormat);
        response = await axios.post(`${API_BASE}/separate?${params.toString()}`, formData, { signal });
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
        if (isTrimming) {
          params.set("trim_start", trimStart);
          params.set("trim_end", trimEnd);
        }
        params.set("export_format", actualFormat);
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
        if (isTrimming) {
          params.set("trim_start", trimStart);
          params.set("trim_end", trimEnd);
        }
        params.set("export_format", actualFormat);

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
        const params = new URLSearchParams({ steps: String(pitchSteps) });
        if (isTrimming) {
          params.set("trim_start", trimStart);
          params.set("trim_end", trimEnd);
        }
        params.set("export_format", actualFormat);
        response = await axios.post(`${API_BASE}/pitch-shift?${params.toString()}`, formData, {
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
        setDownloadFileName(`${baseName}${suffix}.${actualFormat}`);
      }

      if (overrideFormat) {
        setExportFormat(actualFormat);
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

  const handleSingleExport = async (format: string) => {
    if (format === exportFormat && downloadUrl && downloadFileName) {
      // If same format, just download the existing blob
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = downloadFileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setIsSingleExportModalOpen(false);
    } else {
      // If different format, we must re-process the file
      setIsExporting(true);
      await handleUpload(format);
      setIsExporting(false);
      setIsSingleExportModalOpen(false);
      // Let the user manually click download again once done, or we could trigger it automatically 
      // but the UX is fine to just close the modal and they see the new waveform.
      // Alternatively, we can auto-trigger it if we had the new URL here, but state updates are async.
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

            {/* ===== PROCESSING MODULE – redesigned studio console style ===== */}
            <div className="rounded-xl border border-[#1E1E1E] bg-[#0E0E0E] shadow-xl overflow-hidden">

              {/* ── หัว Module พร้อม indicator LED ── */}
              <div className="flex items-center gap-2.5 px-4 pt-4 pb-3 border-b border-[#1E1E1E]">
                {/* LED indicator กะพริบเมื่อกำลังประมวลผล */}
                <span className={`w-2 h-2 rounded-full flex-shrink-0 transition-all duration-300 ${
                  loading
                    ? "bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.6)] animate-pulse"
                    : action === "separate" ? "bg-[#A78BFA] shadow-[0_0_5px_rgba(167,139,250,0.5)]"
                    : action === "eq-ai" ? "bg-[#22D3EE] shadow-[0_0_5px_rgba(34,211,238,0.5)]"
                    : action === "compressor" ? "bg-[#E5A93D] shadow-[0_0_5px_rgba(229,169,61,0.5)]"
                    : "bg-[#34D399] shadow-[0_0_5px_rgba(52,211,153,0.5)]"
                }`} />
                <p className="text-[10px] font-semibold text-[#555555] uppercase tracking-[0.15em]">Processing Module</p>
                <div className="ml-auto text-[10px] font-mono text-[#333333]">
                  {loading ? "RUNNING" : "STANDBY"}
                </div>
              </div>

              <div className="p-4 space-y-4">

                {/* ── Tab selector: console-style ── */}
                <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-[#080808] border border-[#1A1A1A]">
                  {([
                    { value: "separate", label: "Stem Separation", icon: "⊗", activeColor: "text-[#A78BFA] border-[#A78BFA]/60 bg-[#A78BFA]/10 shadow-[0_0_12px_rgba(167,139,250,0.15)]" },
                    { value: "eq-ai",    label: "Auto EQ (AI)",    icon: "⟿", activeColor: "text-[#22D3EE] border-[#22D3EE]/60 bg-[#22D3EE]/10 shadow-[0_0_12px_rgba(34,211,238,0.15)]" },
                    { value: "compressor",label: "Compressor",     icon: "◉", activeColor: "text-[#E5A93D] border-[#E5A93D]/60 bg-[#E5A93D]/10 shadow-[0_0_12px_rgba(229,169,61,0.15)]" },
                    { value: "pitch",    label: "Pitch Shift",     icon: "♯", activeColor: "text-[#34D399] border-[#34D399]/60 bg-[#34D399]/10 shadow-[0_0_12px_rgba(52,211,153,0.15)]" },
                  ] as const).map((item) => (
                    <button
                      key={item.value}
                      onClick={() => setAction(item.value)}
                      disabled={loading}
                      className={`relative flex flex-col items-center gap-0.5 rounded-md px-2 py-2.5 text-xs font-medium border transition-all duration-200 cursor-pointer ${
                        action === item.value
                          ? item.activeColor
                          : "text-[#444444] border-transparent bg-transparent hover:text-[#888888] hover:bg-[#111111]"
                      } disabled:opacity-40 disabled:cursor-not-allowed`}
                    >
                      <span className={`text-base leading-none transition-all duration-200 ${
                        action === item.value ? "opacity-100" : "opacity-40"
                      }`}>{item.icon}</span>
                      <span className="leading-tight text-center">{item.label}</span>
                    </button>
                  ))}
                </div>

                {/* ── Panel ตั้งค่าแต่ละ Module ── */}
                {/* Stem Separation: ไม่มี settings เพิ่มเติม แสดง info card */}
                {action === "separate" && (
                  <div className="rounded-lg border border-[#A78BFA]/20 bg-[#A78BFA]/5 p-3 space-y-2">
                    <p className="text-xs text-[#A78BFA] font-medium">Stem Separation</p>
                    <p className="text-[11px] text-[#666666] leading-relaxed">
                      แยกไฟล์เสียงออกเป็น 4 แทร็กอิสระ — Vocals, Drums, Bass, Other — พร้อมเล่นและ Mix ได้ทันที
                    </p>
                    <div className="grid grid-cols-4 gap-1 pt-1">
                      {["Vocals","Drums","Bass","Other"].map((s) => (
                        <div key={s} className="rounded bg-[#A78BFA]/10 border border-[#A78BFA]/20 py-1 text-center text-[10px] text-[#A78BFA]/80">{s}</div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Genre Profile (เฉพาะ EQ / Compressor) */}
                {(action === "eq-ai" || action === "compressor") && (
                  <div>
                    <label className="block text-[10px] font-semibold text-[#555555] uppercase tracking-[0.12em] mb-1.5">Genre Profile</label>
                    <select
                      value={genre}
                      onChange={(e) => setGenre(e.target.value)}
                      className={`w-full rounded-lg bg-[#080808] border p-2.5 text-[#C8C8C8] text-sm focus:outline-none transition ${
                        action === "eq-ai"
                          ? "border-[#22D3EE]/30 focus:border-[#22D3EE]/70"
                          : "border-[#E5A93D]/30 focus:border-[#E5A93D]/70"
                      }`}
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

                {/* Auto EQ Settings */}
                {action === "eq-ai" && (
                  <div className="rounded-lg border border-[#22D3EE]/20 bg-[#22D3EE]/5 p-3">
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
                  </div>
                )}

                {/* Compressor Settings */}
                {action === "compressor" && (
                  <div className="rounded-lg border border-[#E5A93D]/20 bg-[#E5A93D]/5 p-3">
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
                  </div>
                )}

                {/* Pitch Shift Settings */}
                {action === "pitch" && (
                  <div className="rounded-lg border border-[#34D399]/20 bg-[#34D399]/5 p-3 space-y-3">
                    <PitchShiftSettings
                      pitchSteps={pitchSteps}
                      setPitchSteps={setPitchSteps}
                      loading={loading}
                    />
                  </div>
                )}

                {/* ── Trimming ── */}
                <div className="rounded-lg border border-[#1A1A1A] bg-[#080808] p-3 space-y-3">
                  <label className="flex items-center gap-2.5 cursor-pointer select-none group">
                    {/* Custom checkbox */}
                    <span className={`w-4 h-4 rounded flex-shrink-0 border transition-all duration-150 flex items-center justify-center ${
                      isTrimming ? "bg-[#E5A93D] border-[#E5A93D]" : "border-[#333333] bg-transparent group-hover:border-[#555555]"
                    }`}
                      onClick={() => !loading && setIsTrimming(!isTrimming)}
                    >
                      {isTrimming && (
                        <svg className="w-2.5 h-2.5 text-[#0A0A0A]" viewBox="0 0 10 10" fill="none">
                          <path d="M1.5 5L4 7.5L8.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </span>
                    <input type="checkbox" checked={isTrimming} onChange={(e) => setIsTrimming(e.target.checked)} disabled={loading} className="sr-only" />
                    <span className="text-xs font-medium text-[#888888] group-hover:text-[#AAAAAA] transition-colors">Trim — ตัดช่วงเวลาก่อนประมวลผล</span>
                  </label>
                  {isTrimming && (
                    <div className="grid grid-cols-2 gap-2 pl-6">
                      <div>
                        <label className="block text-[10px] text-[#555555] mb-1">เริ่ม (วินาที)</label>
                        <input
                          type="number" min="0" value={trimStart}
                          onChange={(e) => setTrimStart(e.target.value)} disabled={loading}
                          className="w-full rounded-md bg-[#111111] border border-[#2A2A2A] px-2.5 py-1.5 text-sm text-[#E0E0E0] focus:border-[#E5A93D]/50 focus:outline-none transition"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-[#555555] mb-1">สิ้นสุด (วินาที)</label>
                        <input
                          type="number" min="1" value={trimEnd}
                          onChange={(e) => setTrimEnd(e.target.value)} disabled={loading}
                          className="w-full rounded-md bg-[#111111] border border-[#2A2A2A] px-2.5 py-1.5 text-sm text-[#E0E0E0] focus:border-[#E5A93D]/50 focus:outline-none transition"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* ── Export Format – custom pill toggle ── */}
                <div className="space-y-1.5">
                  <p className="text-[10px] font-semibold text-[#555555] uppercase tracking-[0.12em]">Export Format</p>
                  <div className="grid grid-cols-2 gap-1.5 p-1 rounded-lg bg-[#080808] border border-[#1A1A1A]">
                    {([{ value: "wav", label: "WAV", sub: "Lossless" }, { value: "mp3", label: "MP3", sub: "320 kbps" }] as const).map((fmt) => (
                      <button
                        key={fmt.value}
                        onClick={() => setExportFormat(fmt.value)}
                        className={`flex flex-col items-center py-2 rounded-md text-xs font-medium border transition-all duration-150 cursor-pointer ${
                          exportFormat === fmt.value
                            ? "bg-[#1A1A1A] border-[#E5A93D]/50 text-[#E5A93D] shadow-[0_0_8px_rgba(229,169,61,0.12)]"
                            : "border-transparent text-[#444444] hover:text-[#777777]"
                        }`}
                      >
                        <span className="font-bold text-sm">{fmt.label}</span>
                        <span className="text-[10px] opacity-60 mt-0.5">{fmt.sub}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* ── Execute Button ── */}
                {(() => {
                  // สีและ shadow ของปุ่มเปลี่ยนตาม module ที่เลือก
                  type ActionKey = "separate" | "eq-ai" | "compressor" | "pitch";
                  const moduleStyles: Record<ActionKey, { base: string; glow: string; label: string }> = {
                    separate:   { base: "from-[#7C3AED] to-[#A78BFA]", glow: "rgba(167,139,250,0.35)", label: "Run Stem Separation" },
                    "eq-ai":    { base: "from-[#0891B2] to-[#22D3EE]", glow: "rgba(34,211,238,0.35)",  label: "Apply Auto EQ" },
                    compressor: { base: "from-[#B45309] to-[#E5A93D]", glow: "rgba(229,169,61,0.35)",  label: "Apply Compressor" },
                    pitch:      { base: "from-[#059669] to-[#34D399]", glow: "rgba(52,211,153,0.35)",  label: "Shift Pitch" },
                  };
                  const moduleStyle = moduleStyles[action as ActionKey] ?? moduleStyles.compressor;
                  const isDisabled = loading || (action === "eq-ai" && !isEqDeltaClampValid);
                  return (
                    <button
                      onClick={handleUpload}
                      disabled={isDisabled}
                      className={`relative w-full rounded-lg py-3.5 text-sm font-bold tracking-widest uppercase overflow-hidden transition-all duration-300 ${
                        isDisabled
                          ? "bg-[#1A1A1A] text-[#333333] cursor-not-allowed border border-[#222222]"
                          : `bg-gradient-to-r ${moduleStyle.base} text-white cursor-pointer`
                      }`}
                      style={!isDisabled ? { boxShadow: loading ? `0 0 20px ${moduleStyle.glow}` : `0 0 10px ${moduleStyle.glow}` } : {}}
                    >
                      {/* shimmer overlay เมื่อ processing */}
                      {loading && (
                        <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent animate-[shimmer_1.5s_linear_infinite]" style={{backgroundSize: "200% 100%"}} />
                      )}
                      <span className="relative flex items-center justify-center gap-2">
                        {loading ? (
                          <>
                            <svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.3"/>
                              <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round"/>
                            </svg>
                            Processing…
                          </>
                        ) : moduleStyle.label}
                      </span>
                    </button>
                  );
                })()}

                {/* ── Progress + Status ── */}
                <div className="space-y-2">
                  {/* Segmented meter bar */}
                  <div className="flex gap-px h-1.5 w-full overflow-hidden rounded-full bg-[#111111] border border-[#1A1A1A]">
                    {Array.from({ length: 20 }).map((_, i) => {
                      // แต่ละ segment จะสว่างถ้า progress เกิน threshold
                      const threshold = ((i + 1) / 20) * 100;
                      const lit = progress >= threshold;
                      const segColor = action === "separate" ? "bg-[#A78BFA]"
                        : action === "eq-ai" ? "bg-[#22D3EE]"
                        : action === "compressor" ? "bg-[#E5A93D]"
                        : "bg-[#34D399]";
                      return (
                        <div
                          key={i}
                          className={`flex-1 transition-all duration-150 ${
                            lit ? `${segColor} ${i > 15 ? "opacity-100" : i > 10 ? "opacity-90" : "opacity-80"}` : "bg-[#1C1C1C]"
                          }`}
                        />
                      );
                    })}
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-[11px] text-[#555555]">{statusText || (loading ? "Processing…" : "Ready")}</span>
                    <span className="text-[11px] font-mono text-[#444444]">{progress}%</span>
                  </div>
                  {processingTime && (
                    <div className="text-[11px] text-[#444444] font-mono">⏱ {processingTime}</div>
                  )}
                </div>

                {/* ── Error / Success ── */}
                {errorMessage && (
                  <div className="flex items-start gap-2 rounded-lg border border-red-900/40 bg-red-950/30 px-3 py-2.5">
                    <svg className="w-3.5 h-3.5 text-red-400 flex-shrink-0 mt-0.5" viewBox="0 0 16 16" fill="currentColor">
                      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 10.5a.75.75 0 1 1 0-1.5.75.75 0 0 1 0 1.5zm.75-3.25a.75.75 0 0 1-1.5 0v-3a.75.75 0 0 1 1.5 0v3z"/>
                    </svg>
                    <span className="text-xs text-red-400 leading-snug">{errorMessage}</span>
                  </div>
                )}
                {successMessage && (
                  <div className={`flex items-start gap-2 rounded-lg border px-3 py-2.5 ${
                    action === "separate" ? "border-[#A78BFA]/30 bg-[#A78BFA]/10"
                    : action === "eq-ai" ? "border-[#22D3EE]/30 bg-[#22D3EE]/10"
                    : action === "compressor" ? "border-[#E5A93D]/30 bg-[#E5A93D]/10"
                    : "border-[#34D399]/30 bg-[#34D399]/10"
                  }`}>
                    <svg className={`w-3.5 h-3.5 flex-shrink-0 mt-0.5 ${
                      action === "separate" ? "text-[#A78BFA]"
                      : action === "eq-ai" ? "text-[#22D3EE]"
                      : action === "compressor" ? "text-[#E5A93D]"
                      : "text-[#34D399]"
                    }`} viewBox="0 0 16 16" fill="currentColor">
                      <path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm3.28 5.28-4 4a.75.75 0 0 1-1.06 0l-2-2a.75.75 0 1 1 1.06-1.06l1.47 1.47 3.47-3.47a.75.75 0 0 1 1.06 1.06z"/>
                    </svg>
                    <span className={`text-xs leading-snug ${
                      action === "separate" ? "text-[#A78BFA]"
                      : action === "eq-ai" ? "text-[#22D3EE]"
                      : action === "compressor" ? "text-[#E5A93D]"
                      : "text-[#34D399]"
                    }`}>{successMessage}</span>
                  </div>
                )}

              </div>
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
              <div>
                <MultiStemLivePlayer fileId={fileId} />
              </div>
            )}

            {!loading && zipUrl && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {fileId && (
                  <button
                    onClick={() => setIsExportModalOpen(true)}
                    className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0A0A0A] shadow-[0_4px_15px_rgba(229,169,61,0.2)] transition-all hover:shadow-[0_6px_25px_rgba(229,169,61,0.35)] hover:from-[#F3C05D] hover:to-[#E5A93D]"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    Export & Download
                  </button>
                )}
                {fileId && (
                  <a
                    href={`${API_BASE}/karaoke/${fileId}?export_format=${exportFormat}`}
                    download={`karaoke.${exportFormat}`}
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-[#2A2A2A] bg-[#121212] px-4 py-3.5 font-semibold text-white shadow-[0_4px_15px_rgba(0,0,0,0.2)] transition-all hover:border-[#E5A93D]/50 hover:text-[#E5A93D] hover:bg-[#1A1A1A]"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 opacity-70" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.938l3-8V5a1 1 0 00-1-1H4a1 1 0 00-1 1v1.938l3 8V17a3 3 0 006 0v-2.062z" clipRule="evenodd" />
                    </svg>
                    Download Karaoke
                  </a>
                )}
              </div>
            )}

            <ExportMasterModal 
              isOpen={isExportModalOpen} 
              onClose={() => setIsExportModalOpen(false)} 
              onExport={handleExport}
              isExporting={isExporting}
            />

            <SingleExportModal
              isOpen={isSingleExportModalOpen}
              onClose={() => setIsSingleExportModalOpen(false)}
              onExport={handleSingleExport}
              isExporting={isExporting}
              currentFormat={exportFormat}
            />

            {!loading && downloadUrl && downloadFileName && !downloadFileName.endsWith(".zip") && (
              <div className="rounded-2xl border border-[#222] bg-[#0A0A0A] p-5 space-y-5 shadow-[0_10px_40px_rgba(0,0,0,0.5)]">
                <div>
                  <WaveformPlayer audioUrl={downloadUrl} />
                </div>
                <button
                  onClick={() => setIsSingleExportModalOpen(true)}
                  className="flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl bg-gradient-to-br from-[#E5A93D] to-[#D6962A] px-4 py-3.5 font-bold text-[#0A0A0A] shadow-[0_4px_15px_rgba(229,169,61,0.2)] transition-all hover:shadow-[0_6px_25px_rgba(229,169,61,0.35)] hover:from-[#F3C05D] hover:to-[#E5A93D]"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                  Export & Download
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default UploadBox;
