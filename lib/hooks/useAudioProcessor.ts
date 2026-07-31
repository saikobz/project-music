// lib/hooks/useAudioProcessor.ts
// C7: แยก business logic ทั้งหมดของ UploadBox ออกมาเป็น hook
// (อัปโหลด/ประมวลผล/export/quota/history) — component เหลือแค่ UI

import { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";
import { toast } from "sonner";
import { useSession } from "next-auth/react";
import { API_BASE_URL, ACTION_TO_BACKEND, type AudioAction } from "@/lib/config";
import { downloadViaBlob } from "@/lib/download";
import type { CompressorParams } from "@/app/components/settings/CompressorSettings";

const API_BASE = API_BASE_URL;

// อ่าน X-File-Id ที่ backend ส่งกลับมา (ใช้เก็บ fileId ในประวัติสำหรับ EQ/Compressor/Pitch)
function extractFileId(headers: any): string | undefined {
  if (!headers) return undefined;
  const value =
    typeof headers.get === "function" ? headers.get("x-file-id") : headers["x-file-id"] ?? headers["X-File-Id"];
  return typeof value === "string" && value ? value : undefined;
}

export interface AudioAnalysisResult {
  tempo: number;
  key: string;
  pitch: string | null;
}

export interface UseAudioProcessorInput {
  file: File | null;
  action: AudioAction;
  exportFormat: string;
  setExportFormat: (format: string) => void;
  isTrimming: boolean;
  trimStart: string;
  trimEnd: string;
  genre: string;
  autoEqModel: string;
  deltaClampDb: string;
  compParams: CompressorParams;
  pitchSteps: number;
}

export function useAudioProcessor({
  file,
  action,
  exportFormat,
  setExportFormat,
  isTrimming,
  trimStart,
  trimEnd,
  genre,
  autoEqModel,
  deltaClampDb,
  compParams,
  pitchSteps,
}: UseAudioProcessorInput) {
  const { data: session } = useSession();
  const userTier = (session?.user as any)?.tier || "FREE";

  // สถานะผลลัพธ์จาก backend
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const downloadUrlRef = useRef<string | null>(null);
  useEffect(() => {
    downloadUrlRef.current = downloadUrl;
  }, [downloadUrl]);
  const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [zipUrl, setZipUrl] = useState<string | null>(null);

  // ใช้กับ SingleExportModal: รอ blob URL ใหม่แล้วดาวน์โหลดอัตโนมัติ (F12)
  const [pendingAutoDownload, setPendingAutoDownload] = useState(false);
  useEffect(() => {
    if (pendingAutoDownload && downloadUrl && downloadFileName) {
      setPendingAutoDownload(false);
      const a = document.createElement("a");
      a.href = downloadUrl;
      a.download = downloadFileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [pendingAutoDownload, downloadUrl, downloadFileName]);

  // สถานะ modal
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isSingleExportModalOpen, setIsSingleExportModalOpen] = useState(false);

  // สถานะ UI
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AudioAnalysisResult | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // abort request ค้างไว้ตอน unmount + revoke blob URL
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      if (downloadUrlRef.current) {
        URL.revokeObjectURL(downloadUrlRef.current);
        downloadUrlRef.current = null;
      }
    };
  }, []);

  // ล้าง blob URL เก่าทุกครั้ง (M7)
  const clearDownloadUrl = useCallback(() => {
    if (downloadUrlRef.current) {
      URL.revokeObjectURL(downloadUrlRef.current);
      downloadUrlRef.current = null;
    }
    setDownloadUrl(null);
  }, []);

  const saveHistory = useCallback(
    (actionName: AudioAction, historyFileId?: string, stems?: string[]) => {
      if (!session) {
        console.log("saveHistory: skipped (not logged in)");
        return;
      }
      fetch("/api/history", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: ACTION_TO_BACKEND[actionName] || actionName,
          originalFilename: file?.name || "unknown.wav",
          ...(historyFileId && { fileId: historyFileId }),
          ...(stems && { stems }),
        }),
      }).catch((err) => console.error("saveHistory failed:", err));
    },
    [session, file]
  );

  const handleExport = useCallback(
    async (exportType: string, format: string, targetLufs: number, selectedStems: string[]) => {
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
    },
    [fileId]
  );

  const handleUpload = useCallback(
    async (overrideFormat?: string | React.MouseEvent | React.FormEvent) => {
      if (!file) {
        setErrorMessage("กรุณาเลือกไฟล์ WAV (≤100MB) ก่อนเริ่มประมวลผล");
        return;
      }

      const actualFormat = typeof overrideFormat === "string" ? overrideFormat : exportFormat;

      // M14: ตรวจสอบช่วงเวลา trim ก่อน submit (backend ก็ validate ไว้แล้ว แต่ block เร็วเพื่อ UX)
      if (isTrimming) {
        const trimStartNum = Number.parseFloat(trimStart);
        const trimEndNum = Number.parseFloat(trimEnd);
        if (!Number.isFinite(trimStartNum) || trimStartNum < 0) {
          setErrorMessage("ค่าเริ่มต้น (trim_start) ต้องเป็นตัวเลขที่ไม่ติดลบ");
          toast.error("ค่าเริ่มต้น (trim_start) ต้องเป็นตัวเลขที่ไม่ติดลบ");
          return;
        }
        if (!Number.isFinite(trimEndNum) || trimEndNum <= trimStartNum) {
          setErrorMessage("ค่าสิ้นสุด (trim_end) ต้องมากกว่าค่าเริ่มต้น");
          toast.error("ค่าสิ้นสุด (trim_end) ต้องมากกว่าค่าเริ่มต้น");
          return;
        }
      }

      setLoading(true);
      clearDownloadUrl();
      setDownloadFileName(null);
      setProcessingTime(null);
      setFileId(null);
      setZipUrl(null);
      setErrorMessage(null);
      setSuccessMessage(null);
      setAnalysis(null);
      setStatusText("กำลังอัปโหลดและประมวลผล...");
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();
      const signal = abortControllerRef.current.signal;

      const formData = new FormData();
      formData.append("file", file);

      const startTime = Date.now();

      // กันการ refund ซ้ำซ้อน: เฉพาะงานที่หักโควตาไปแล้ว (และไม่ได้ถูกยกเลิก) ถึงจะคืน
      let quotaCharged = false;

      try {
        let response: any;
        let suffix = "";
        let successMsg = "ประมวลผลเสร็จแล้ว";
        const userId = (session?.user as any)?.id;
        const reqHeaders: Record<string, string> = { "X-User-Tier": userTier };
        if (userId) {
          reqHeaders["X-User-Id"] = userId;
        }

        // สำหรับผู้ใช้ที่ Login แล้ว: ตรวจสอบและนับโควตาจาก Database ก่อนส่งงานไปประมวลผล
        if (userId) {
          const quotaRes = await fetch("/api/quota/consume", { method: "POST" });
          if (quotaRes.status === 403) {
            const quotaErr = await quotaRes.json().catch(() => null);
            throw new Error(quotaErr?.error || "โควตาประมวลผลฟรีเต็มแล้ว กรุณาสมัครสมาชิกเพื่อใช้งานต่อ");
          }
          if (!quotaRes.ok) {
            console.error(`Quota check failed (${quotaRes.status}), continuing without tracking`);
          } else {
            quotaCharged = true;
          }
        }

        if (action === "separate") {
          const params = new URLSearchParams();
          if (isTrimming) {
            params.set("trim_start", trimStart);
            params.set("trim_end", trimEnd);
          }
          params.set("export_format", exportFormat);
          response = await axios.post(`${API_BASE}/separate?${params.toString()}`, formData, {
            signal,
            headers: reqHeaders,
          });
          const { file_id, zip_url } = response.data;
          setFileId(file_id);
          setZipUrl(zip_url);
          successMsg = "แยกเสียงเสร็จแล้ว ดาวน์โหลด ZIP หรือลองเล่นทีละสเตมได้เลย";
          setStatusText("กำลังเตรียมไฟล์สเตม...");
          saveHistory("separate", file_id, ["Vocals", "Drums", "Bass", "Other"]);
        }

        if (action === "eq-ai") {
          const params = new URLSearchParams({
            genre,
            model_id: autoEqModel,
            delta_clamp_db: deltaClampDb || "2",
          });
          if (isTrimming) {
            params.set("trim_start", trimStart);
            params.set("trim_end", trimEnd);
          }
          params.set("export_format", actualFormat);
          response = await axios.post(`${API_BASE}/apply-eq-ai?${params.toString()}`, formData, {
            responseType: "blob",
            signal,
            headers: reqHeaders,
          });
          const url = window.URL.createObjectURL(new Blob([response.data]));
          setDownloadUrl(url);
          suffix = `_eq_ai_${autoEqModel}_${genre}_${deltaClampDb}db`;
          successMsg = "Auto-EQ ประมวลผลเสร็จสิ้น";
          setStatusText("Running Auto-EQ...");
          saveHistory("eq-ai", extractFileId(response.headers));
        }

        if (action === "compressor") {
          const params = new URLSearchParams({
            strength: compParams.strength,
            genre,
            knee: compParams.knee || "6",
            makeup_gain: compParams.makeupGain || "0",
            dry_wet: compParams.dryWet || "100",
          });
          if (compParams.threshold) params.set("threshold", compParams.threshold);
          if (compParams.ratio) params.set("ratio", compParams.ratio);
          if (compParams.attack) params.set("attack", compParams.attack);
          if (compParams.release) params.set("release", compParams.release);
          if (compParams.outputCeiling) params.set("output_ceiling", compParams.outputCeiling);
          if (isTrimming) {
            params.set("trim_start", trimStart);
            params.set("trim_end", trimEnd);
          }
          params.set("export_format", actualFormat);

          response = await axios.post(`${API_BASE}/apply-compressor?${params.toString()}`, formData, {
            responseType: "blob",
            signal,
            headers: reqHeaders,
          });
          const url = window.URL.createObjectURL(new Blob([response.data]));
          setDownloadUrl(url);
          suffix = `_compressed_${compParams.strength}`;
          successMsg = "ประมวลผล Compressor เสร็จแล้ว";
          setStatusText("กำลังสร้างไฟล์ Compressor...");
          saveHistory("compressor", extractFileId(response.headers));
        }

        if (action === "pitch") {
          const params = new URLSearchParams({ steps: String(pitchSteps) });
          if (isTrimming) {
            params.set("trim_start", trimStart);
            params.set("trim_end", trimEnd);
          }
          params.set("export_format", actualFormat);
          response = await axios.post(`${API_BASE}/pitch-shift?${params.toString()}`, formData, {
            responseType: "blob",
            signal,
            headers: reqHeaders,
          });
          const url = window.URL.createObjectURL(new Blob([response.data]));
          setDownloadUrl(url);
          suffix = `_pitch_${pitchSteps}`;
          successMsg = "ประมวลผล Pitch Shift เสร็จแล้ว";
          setStatusText("กำลังสร้างไฟล์ Pitch Shift...");
          saveHistory("pitch", extractFileId(response.headers));
        }

        if (file && suffix) {
          const baseName = file.name.replace(/\.[^/.]+$/, "");
          setDownloadFileName(`${baseName}${suffix}.${actualFormat}`);
        }

        if (overrideFormat) {
          setExportFormat(actualFormat);
        }

        const analyzeData = new FormData();
        analyzeData.append("file", file);
        try {
          const analyzeResp = await axios.post(`${API_BASE}/analyze`, analyzeData, {
            signal,
            headers: reqHeaders,
          });
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
        setStatusText("เสร็จแล้ว! ดาวน์โหลดหรือเล่นไฟล์ได้เลย");
        setSuccessMessage(successMsg);
        toast.success(successMsg);
      } catch (err: any) {
        if (axios.isCancel(err)) {
          // ผู้ใช้ยกเลิกกลางคัน -> คืนโควตาที่หักไป (ยังไม่ได้ประมวลผลสำเร็จ)
          if (quotaCharged) {
            fetch("/api/quota/refund", { method: "POST" }).catch(() => {
              console.error("Failed to refund quota on cancel");
            });
          }
          console.log("Request canceled by user or refresh");
          return;
        }
        // ประมวลผลล้มเหลว -> คืนโควตาที่หักไป (F11)
        if (quotaCharged) {
          fetch("/api/quota/refund", { method: "POST" }).catch(() => {
            console.error("Failed to refund quota");
          });
        }
        let message = "เกิดข้อผิดพลาดระหว่างประมวลผล กรุณาลองใหม่";

        if (err?.response?.data) {
          if (err.response.data instanceof Blob) {
            try {
              const text = await err.response.data.text();
              const parsed = JSON.parse(text);
              message = parsed.detail || parsed.message || text;
            } catch {
              message = err.message || `คำขอไม่สำเร็จ (${err.response.status})`;
            }
          } else if (typeof err.response.data === "string") {
            message = err.response.data;
          } else if (err.response.data.detail) {
            message = err.response.data.detail;
          } else if (err.response.data.message) {
            message = err.response.data.message;
          }
        } else if (err.code === "ERR_NETWORK") {
          message = "ติดต่อ backend ไม่ได้ (ตรวจสอบการรันเซิร์ฟเวอร์หรือ CORS)";
        } else if (err.message) {
          message = err.message;
        }

        setErrorMessage(message);
        toast.error(message);
        setStatusText(null);
      } finally {
        setLoading(false);
      }
    },
    [
      file,
      exportFormat,
      setExportFormat,
      isTrimming,
      trimStart,
      trimEnd,
      action,
      genre,
      autoEqModel,
      deltaClampDb,
      compParams,
      pitchSteps,
      session,
      userTier,
      clearDownloadUrl,
      saveHistory,
    ]
  );

  const handleSingleExport = useCallback(
    async (format: string) => {
      if (format === exportFormat && downloadUrl && downloadFileName) {
        // If same format, just download the existing blob
        const a = document.createElement("a");
        a.href = downloadUrl;
        a.download = downloadFileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setIsSingleExportModalOpen(false);
      } else if (downloadUrl && downloadFileName) {
        // F12: format ต่าง -> แปลงไฟล์ที่ประมวลผลแล้วที่ backend (/convert-format)
        setIsExporting(true);
        try {
          const blobRes = await fetch(downloadUrl);
          const processedBlob = await blobRes.blob();
          const formData = new FormData();
          formData.append("file", processedBlob, downloadFileName);

          const convertRes = await axios.post(
            `${API_BASE}/convert-format?export_format=${format}`,
            formData,
            {
              responseType: "blob",
              signal: abortControllerRef.current?.signal,
            }
          );

          const newUrl = URL.createObjectURL(new Blob([convertRes.data]));
          clearDownloadUrl();
          setDownloadUrl(newUrl);
          downloadUrlRef.current = newUrl;
          setDownloadFileName((prev) =>
            prev ? prev.replace(/\.[^/.]+$/, `.${format}`) : `output.${format}`
          );
          setIsSingleExportModalOpen(false);
          setPendingAutoDownload(true);
        } catch (err) {
          console.error("Format conversion failed:", err);
          toast.error("ไม่สามารถแปลงฟอร์แมตเสียงได้ กรุณาลองใหม่");
        } finally {
          setIsExporting(false);
        }
      }
    },
    [exportFormat, downloadUrl, downloadFileName, clearDownloadUrl]
  );

  const handleKaraokeDownload = useCallback(async () => {
    if (!fileId) return;
    // M9: ดาวน์โหลดผ่าน fetch -> blob (ลิงก์ข้าม origin + download attribute ไม่ทำงาน)
    const ok = await downloadViaBlob(
      `${API_BASE}/karaoke/${fileId}?export_format=${exportFormat}`,
      `karaoke.${exportFormat}`
    );
    if (!ok) {
      toast.error("ไม่สามารถดาวน์โหลดคาราโอเกะได้ (ไฟล์อาจหมดอายุแล้ว) กรุณาประมวลผลใหม่");
    }
  }, [fileId, exportFormat]);

  return {
    // สถานะผลลัพธ์
    loading,
    errorMessage,
    successMessage,
    statusText,
    analysis,
    downloadUrl,
    downloadFileName,
    processingTime,
    fileId,
    zipUrl,
    // สถานะ modal
    isExportModalOpen,
    setIsExportModalOpen,
    isExporting,
    isSingleExportModalOpen,
    setIsSingleExportModalOpen,
    // setter ที่ component ต้องใช้ (เช่น handleFileSelect ล้างผลลัพธ์)
    setErrorMessage,
    setSuccessMessage,
    setAnalysis,
    setDownloadFileName,
    setFileId,
    setZipUrl,
    setProcessingTime,
    clearDownloadUrl,
    // ฟังก์ชันหลัก
    handleUpload,
    handleExport,
    handleSingleExport,
    handleKaraokeDownload,
    saveHistory,
  };
}
