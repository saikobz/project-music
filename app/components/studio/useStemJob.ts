"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { useSession } from "next-auth/react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const MAX_SIZE_BYTES = 100 * 1024 * 1024; // 100MB

export interface UseStemJobOptions {
  onHeightChange?: (expanded: boolean) => void;
}

export function useStemJob(options?: UseStemJobOptions) {
  const { data: session } = useSession();
  const userTier = (session?.user as any)?.tier || "FREE";
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [action, setAction] = useState("separate");
  const [strength, setStrength] = useState("medium");
  const [genre, setGenre] = useState("pop");
  const [autoEqModel, setAutoEqModel] = useState("lstm-last");
  const [deltaClampDb, setDeltaClampDb] = useState("2");
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

  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

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

  const hasResults = Boolean(fileId || downloadUrl || analysis);

  useEffect(() => {
    options?.onHeightChange?.(hasResults);
  }, [hasResults, options]);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const validateAndSetFile = (selectedFile: File) => {
    const isWavExt = selectedFile.name.toLowerCase().endsWith(".wav");
    const isWavMime = selectedFile.type === "audio/wav" || selectedFile.type === "audio/x-wav";

    if (!isWavExt && !isWavMime) {
      const msg = "รองรับเฉพาะไฟล์ WAV (.wav) เท่านั้น";
      setErrorMessage(msg);
      toast.error(msg);
      setFile(null);
      return false;
    }

    if (selectedFile.size > MAX_SIZE_BYTES) {
      const msg = "ไฟล์ต้องมีขนาดไม่เกิน 100MB";
      setErrorMessage(msg);
      toast.error(msg);
      setFile(null);
      return false;
    }

    setFile(selectedFile);
    setErrorMessage(null);
    setSuccessMessage(null);
    setDownloadUrl(null);
    setDownloadFileName(null);
    setFileId(null);
    setZipUrl(null);
    setAnalysis(null);
    return true;
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const startProgressSimulation = () => {
    setProgress(5);
    if (progressTimerRef.current) clearInterval(progressTimerRef.current);
    progressTimerRef.current = setInterval(() => {
      setProgress((prev) => (prev < 90 ? prev + Math.floor(Math.random() * 5) + 2 : prev));
    }, 500);
  };

  const stopProgressSimulation = (finalProgress = 100) => {
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
    setProgress(finalProgress);
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
        method: "POST",
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
    if (!file) {
      setErrorMessage("กรุณาเลือกไฟล์ WAV ก่อนกดประมวลผล");
      toast.error("กรุณาเลือกไฟล์ WAV ก่อนกดประมวลผล");
      return;
    }

    setLoading(true);
    setErrorMessage(null);
    setSuccessMessage(null);
    setDownloadUrl(null);
    setDownloadFileName(null);
    setFileId(null);
    setZipUrl(null);
    setAnalysis(null);
    setStatusText("กำลังอัปโหลดไฟล์ไปยังเซิร์ฟเวอร์...");
    startProgressSimulation();

    const formData = new FormData();
    formData.append("file", file);

    const params: Record<string, any> = {};
    if (isTrimming) {
      if (trimStart !== "") params.trim_start = trimStart;
      if (trimEnd !== "") params.trim_end = trimEnd;
    }

    let endpoint = "";
    if (action === "separate") {
      endpoint = "/separate";
      params.export_format = exportFormat;
    } else if (action === "apply-eq-ai") {
      endpoint = "/apply-eq-ai";
      params.genre = genre;
      params.model_id = autoEqModel;
      params.delta_clamp_db = deltaClampDb;
      params.export_format = exportFormat;
    } else if (action === "apply-compressor") {
      endpoint = "/apply-compressor";
      params.strength = strength;
      params.genre = genre;
      params.export_format = exportFormat;
      if (compThreshold !== "") params.threshold = compThreshold;
      if (compRatio !== "") params.ratio = compRatio;
      if (compAttack !== "") params.attack = compAttack;
      if (compRelease !== "") params.release = compRelease;
      if (compKnee !== "") params.knee = compKnee;
      if (compMakeupGain !== "") params.makeup_gain = compMakeupGain;
      if (compDryWet !== "") params.dry_wet = compDryWet;
      if (compOutputCeiling !== "") params.output_ceiling = compOutputCeiling;
    } else if (action === "pitch-shift") {
      endpoint = "/pitch-shift";
      params.steps = pitchSteps;
      params.export_format = exportFormat;
    } else if (action === "analyze") {
      endpoint = "/analyze";
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const requestOptions = {
        params,
        signal: controller.signal,
        headers: { "X-User-Tier": userTier },
      };

      if (action === "analyze") {
        const response = await axios.post(`${API_BASE}${endpoint}`, formData, requestOptions);
        setAnalysis(response.data);
        stopProgressSimulation(100);
        setSuccessMessage("วิเคราะห์ไฟล์เสียงสำเร็จ!");
        toast.success("วิเคราะห์ไฟล์เสียงสำเร็จ!");
      } else if (action === "separate") {
        const response = await axios.post(`${API_BASE}${endpoint}`, formData, requestOptions);
        const data = response.data;
        if (data.status === "success") {
          setFileId(data.file_id);
          setZipUrl(`${API_BASE}${data.zip_url}`);
          stopProgressSimulation(100);
          setSuccessMessage("แยกเสียงสำเร็จ! สามารถดาวน์โหลดและมิกซ์เสียงได้ทันที");
          toast.success("แยกเสียงสำเร็จ!");
        }
      } else {
        const response = await axios.post(`${API_BASE}${endpoint}`, formData, {
          ...requestOptions,
          responseType: "blob",
        });
        const blob = new Blob([response.data], {
          type: exportFormat === "mp3" ? "audio/mpeg" : "audio/wav",
        });
        const url = URL.createObjectURL(blob);
        setDownloadUrl(url);

        let outExt = exportFormat;
        let outName = `output.${outExt}`;
        if (action === "apply-eq-ai") outName = `eq_${genre}_${autoEqModel}.${outExt}`;
        else if (action === "apply-compressor") outName = `compressed_${strength}.${outExt}`;
        else if (action === "pitch-shift") outName = `pitch_${pitchSteps}semitones.${outExt}`;

        setDownloadFileName(outName);
        stopProgressSimulation(100);
        setSuccessMessage("ประมวลผลเสียงสำเร็จ!");
        toast.success("ประมวลผลเสียงสำเร็จ!");
      }
    } catch (err: any) {
      stopProgressSimulation(0);
      let errMsg = "เกิดข้อผิดพลาดในการประมวลผล";

      if (err.response?.data) {
        if (err.response.data instanceof Blob) {
          try {
            const text = await err.response.data.text();
            const parsed = JSON.parse(text);
            errMsg = parsed.detail || parsed.message || text;
          } catch {
            errMsg = err.message || "เกิดข้อผิดพลาดจากเซิร์ฟเวอร์";
          }
        } else if (typeof err.response.data === "string") {
          errMsg = err.response.data;
        } else if (err.response.data.detail) {
          errMsg = err.response.data.detail;
        } else if (err.response.data.message) {
          errMsg = err.response.data.message;
        }
      } else if (err.message) {
        errMsg = err.message;
      }

      setErrorMessage(errMsg);
      toast.error(errMsg);
    } finally {
      setLoading(false);
      setStatusText(null);
    }
  };

  return {
    file,
    isDragging,
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
    handleDragOver,
    handleDragLeave,
    handleDrop,
    handleFileChange,
    handleSubmit,
    handleExport,
    validateAndSetFile,
  };
}
