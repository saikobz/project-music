// app/components/studio/useAudioProcessor.ts
"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { useSession } from "next-auth/react";
import { API_BASE_URL, MAX_UPLOAD_BYTES } from "@/lib/config";

const API_BASE = API_BASE_URL;
const MAX_SIZE_BYTES = MAX_UPLOAD_BYTES;
const AUTO_EQ_DELTA_CLAMP_DEFAULT = 2;
const AUTO_EQ_MODEL_DEFAULT = "lstm-last";

export function useAudioProcessor(onHeightChange?: (expanded: boolean) => void) {
  const { data: session } = useSession();
  const userTier = (session?.user as any)?.tier || "FREE";

  // Form State
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

  // Output / Result State
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

  // Modal & Status State
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isSingleExportModalOpen, setIsSingleExportModalOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<{ tempo: number; key: string; pitch: string | null } | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const progressTimerRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (onHeightChange) {
      const expanded = action !== "separate" || !!file || loading || !!zipUrl || !!downloadUrl || !!analysis;
      onHeightChange(expanded);
    }
  }, [action, file, loading, zipUrl, downloadUrl, analysis, onHeightChange]);

  useEffect(() => {
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

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (progressTimerRef.current) {
        clearInterval(progressTimerRef.current);
      }
    };
  }, []);

  const handleFileSelect = (selected: File | null) => {
    setErrorMessage(null);
    setSuccessMessage(null);
    setFile(null);
    if (!selected) return;
    const ext = selected.name.toLowerCase().split(".").pop();
    if (ext !== "wav") {
      const msg = "รองรับเฉพาะไฟล์ WAV (.wav) เท่านั้น";
      setErrorMessage(msg);
      toast.error(msg);
      return;
    }
    if (selected.size > MAX_SIZE_BYTES) {
      const msg = "ไฟล์มีขนาดเกิน 100MB กรุณาเลือกไฟล์ที่มีขนาดเล็กลง";
      setErrorMessage(msg);
      toast.error(msg);
      return;
    }
    setFile(selected);
    toast.success(`อัปโหลดไฟล์ ${selected.name} สำเร็จ`);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const startProgressTimer = (estimatedTotalMs = 15000) => {
    setProgress(0);
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
    }
    const intervalMs = 200;
    const increment = (95 / (estimatedTotalMs / intervalMs));
    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) {
          return 95;
        }
        return Math.min(95, prev + increment);
      });
    }, intervalMs);
  };

  const stopProgressTimer = () => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setProgress(100);
  };

  const resetResults = () => {
    setDownloadUrl(null);
    setDownloadFileName(null);
    setProcessingTime(null);
    setErrorMessage(null);
    setSuccessMessage(null);
    setAnalysis(null);
    setZipUrl(null);
    setFileId(null);
    setStatusText(null);
    setProgress(0);
  };

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
        method: "POST"
      });
      if (!res.ok) {
        throw new Error(`เซิร์ฟเวอร์ส่งข้อผิดพลาด (${res.status})`);
      }
      const data = await res.json();
      if (data.status === "success") {
        const url = data.file_url.startsWith("http") ? data.file_url : `${API_BASE}${data.file_url}`;
        const fileRes = await fetch(url);
        if (!fileRes.ok) {
          throw new Error("ดาวน์โหลดไฟล์เสียงที่ประมวลผลแล้วไม่สำเร็จ");
        }
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
        toast.success("ส่งออกไฟล์เสียงและเริ่มดาวน์โหลดสำเร็จ");
      } else {
        throw new Error(data.message || "เกิดข้อผิดพลาดในการส่งออกไฟล์");
      }
    } catch (err: any) {
      console.error("Export failed:", err);
      toast.error(err.message || "การส่งออกไฟล์ล้มเหลว กรุณาลองใหม่อีกครั้ง");
    } finally {
      setIsExporting(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    resetResults();
    setLoading(true);
    const startTime = Date.now();
    startProgressTimer(action === "separate" ? 20000 : 10000);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("export_format", exportFormat);

    if (isTrimming) {
      const startSec = parseFloat(trimStart);
      const endSec = parseFloat(trimEnd);
      if (!isNaN(startSec) && startSec >= 0) {
        formData.append("trim_start", startSec.toString());
      }
      if (!isNaN(endSec) && endSec > 0) {
        formData.append("trim_end", endSec.toString());
      }
    }

    let endpoint = `${API_BASE}/separate`;
    if (action === "auto_eq") {
      endpoint = `${API_BASE}/apply-eq`;
      formData.append("genre", genre);
      formData.append("strength", strength);
      formData.append("model_type", autoEqModel);
      formData.append("delta_clamp_db", deltaClampDb);
    } else if (action === "compressor") {
      endpoint = `${API_BASE}/compress`;
      if (compThreshold !== "") formData.append("threshold_db", compThreshold);
      if (compRatio !== "") formData.append("ratio", compRatio);
      if (compAttack !== "") formData.append("attack_ms", compAttack);
      if (compRelease !== "") formData.append("release_ms", compRelease);
      if (compKnee !== "") formData.append("knee_db", compKnee);
      if (compMakeupGain !== "") formData.append("makeup_gain_db", compMakeupGain);
      if (compDryWet !== "") formData.append("dry_wet_percent", compDryWet);
      if (compOutputCeiling !== "") formData.append("output_ceiling_db", compOutputCeiling);
    } else if (action === "pitch_shift") {
      endpoint = `${API_BASE}/pitch-shift`;
      formData.append("semitones", pitchSteps.toString());
    }

    abortControllerRef.current = new AbortController();

    try {
      setStatusText("กำลังอัปโหลดและประมวลผลไฟล์...");
      const response = await axios.post(endpoint, formData, {
        signal: abortControllerRef.current.signal,
        headers: {
          "x-user-tier": userTier
        }
      });

      stopProgressTimer();
      const elapsedSeconds = ((Date.now() - startTime) / 1000).toFixed(1);
      setProcessingTime(`${elapsedSeconds} วินาที`);

      if (action === "separate") {
        if (response.data.zip_url) {
          setZipUrl(`${API_BASE}${response.data.zip_url}`);
          setFileId(response.data.file_id);
          setSuccessMessage("แยกเสียงสำเร็จแล้ว!");
          toast.success("แยกเสียงดนตรีสำเร็จแล้ว!");
        }
      } else {
        if (response.data.download_url) {
          const absoluteDownloadUrl = `${API_BASE}${response.data.download_url}`;
          setDownloadUrl(absoluteDownloadUrl);
          setFileId(response.data.file_id);
          const baseName = file.name.substring(0, file.name.lastIndexOf(".")) || file.name;
          const ext = exportFormat === "mp3" ? "mp3" : "wav";
          let suffix = "processed";
          if (action === "auto_eq") suffix = `eq_${genre}_${strength}`;
          if (action === "compressor") suffix = "compressed";
          if (action === "pitch_shift") suffix = `pitch_${pitchSteps > 0 ? "+" : ""}${pitchSteps}st`;
          setDownloadFileName(`${baseName}_${suffix}.${ext}`);
        }
        if (response.data.analysis) {
          setAnalysis(response.data.analysis);
        }
        setSuccessMessage("ประมวลผลเสียงสำเร็จแล้ว!");
        toast.success("ประมวลผลเสียงเรียบร้อยแล้ว!");
      }
    } catch (err: any) {
      stopProgressTimer();
      if (axios.isCancel(err)) {
        toast.info("ยกเลิกการประมวลผลแล้ว");
      } else {
        const errorDetail = err.response?.data?.detail || err.message || "เกิดข้อผิดพลาดไม่ทราบสาเหตุ";
        setErrorMessage(errorDetail);
        toast.error(`การประมวลผลล้มเหลว: ${errorDetail}`);
      }
    } finally {
      setLoading(false);
      abortControllerRef.current = null;
    }
  };

  return {
    userTier,
    file,
    isDragging,
    setIsDragging,
    action,
    setAction,
    strength,
    setStrength,
    genre,
    setGenre,
    autoEqModel,
    setAutoEqModel,
    deltaClampDb,
    setDeltaClampDb,
    compThreshold,
    setCompThreshold,
    compRatio,
    setCompRatio,
    compAttack,
    setCompAttack,
    compRelease,
    setCompRelease,
    compKnee,
    setCompKnee,
    compMakeupGain,
    setCompMakeupGain,
    compDryWet,
    setCompDryWet,
    compOutputCeiling,
    setCompOutputCeiling,
    pitchSteps,
    setPitchSteps,
    isTrimming,
    setIsTrimming,
    trimStart,
    setTrimStart,
    trimEnd,
    setTrimEnd,
    exportFormat,
    setExportFormat,
    downloadUrl,
    downloadFileName,
    processingTime,
    fileId,
    zipUrl,
    isExportModalOpen,
    setIsExportModalOpen,
    isExporting,
    isSingleExportModalOpen,
    setIsSingleExportModalOpen,
    loading,
    errorMessage,
    successMessage,
    analysis,
    statusText,
    progress,
    handleFileSelect,
    handleDrop,
    handleExport,
    handleSubmit,
    API_BASE
  };
}
