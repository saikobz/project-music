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
// ที่อยู่ของ backend และข้อจำกัดขนาดไฟล์ฝั่งหน้าเว็บ

// ค่าตั้งต้นของ API และข้อจำกัดการอัปโหลด
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const MAX_SIZE_BYTES = 100 * 1024 * 1024; // 100MB
const AUTO_EQ_DELTA_CLAMP_MIN = 0;
const AUTO_EQ_DELTA_CLAMP_MAX = 6;
const AUTO_EQ_DELTA_CLAMP_DEFAULT = 2;
const AUTO_EQ_MODEL_DEFAULT = "lstm-last";
const AUTO_EQ_MODEL_OPTIONS = [
  { value: "cnn-v1", label: "CNN v1", hint: "โหมดเดิมของโปรเจกต์" },
  { value: "lstm-last", label: "LSTM Last", hint: "โมเดลใหม่แบบ sequence-aware" },
];

function UploadBox() {
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
        response = await axios.post(`${API_BASE}/separate`, formData);
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
    <div className="grid gap-6 md:grid-cols-2 p-6 text-[#EDE9FE]">
      <div className="space-y-4">
        {/* ===== การ์ดเลือกไฟล์เสียง ===== */}
        {/* ขั้นตอนที่ 1: รับไฟล์ WAV ผ่านปุ่มเลือกไฟล์หรือ drag-and-drop */}
        {/* การ์ดเลือกไฟล์ WAV และพื้นที่ drag-and-drop */}
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

        {/* ขั้นตอนที่ 2: แสดงเฉพาะ control ที่จำเป็นสำหรับ action ที่เลือก */}
        {/* การ์ดเลือก action และตั้งค่าพารามิเตอร์ของงานที่เลือก */}
        <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur space-y-3 shadow-inner shadow-purple-900/30">
          {/* ===== การ์ดเลือกประเภทงานและตั้งค่าพารามิเตอร์ ===== */}
          <p className="text-sm text-[#A78BFA] font-semibold">ขั้นตอนที่ 2</p>
          <h3 className="text-xl font-semibold">เลือกระบบที่ต้องการ</h3>
          <div className="grid grid-cols-2 gap-2">
            {[
              { value: "separate", label: "แยกเสียง" },
              { value: "eq-ai", label: "EQ (AI)" },
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

          {/* ตัวเลือก genre ใช้กับ Auto-EQ และ Compressor */}
          {/* ===== ส่วนเลือก genre: ใช้กับ EQ AI และ Compressor ===== */}
          {(action === "eq-ai" || action === "compressor") && (
            <div>
              <label className="block text-sm mb-1">แนวเพลง (Genre)</label>
              <select
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#fee9e9]"
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
            <div className="space-y-2">
              <div>
                <label className="block text-sm mb-1">Auto-EQ Model</label>
                <select
                  value={autoEqModel}
                  onChange={(e) => setAutoEqModel(e.target.value)}
                  className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                  disabled={loading}
                >
                  {AUTO_EQ_MODEL_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-[#A78BFA]">{selectedAutoEqModel.hint}</p>
              </div>
              <div className="flex items-center justify-between gap-3">
                <label className="block text-sm mb-1">Delta Clamp (dB)</label>
                <input
                  type="number"
                  min={AUTO_EQ_DELTA_CLAMP_MIN}
                  max={AUTO_EQ_DELTA_CLAMP_MAX}
                  step="0.1"
                  value={deltaClampDb}
                  onChange={(e) => setDeltaClampDb(e.target.value)}
                  className="w-24 rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-right text-[#EDE9FE]"
                  disabled={loading}
                />
              </div>
              <input
                type="range"
                min={AUTO_EQ_DELTA_CLAMP_MIN}
                max={AUTO_EQ_DELTA_CLAMP_MAX}
                step="0.1"
                value={deltaClampDb}
                onChange={(e) => setDeltaClampDb(e.target.value)}
                className="w-full accent-[#22D3EE]"
                disabled={loading}
              />
              <div className="flex justify-between text-xs text-[#A78BFA]">
                <span>{AUTO_EQ_DELTA_CLAMP_MIN} dB</span>
                <span>ค่าเริ่มต้น {AUTO_EQ_DELTA_CLAMP_DEFAULT} dB</span>
                <span>{AUTO_EQ_DELTA_CLAMP_MAX} dB</span>
              </div>
              <div
                className={`rounded-lg border px-3 py-2 text-xs leading-5 ${
                  isEqDeltaClampValid
                    ? "border-amber-400/40 bg-amber-500/10 text-amber-100"
                    : "border-red-400/50 bg-red-500/10 text-red-100"
                }`}
              >
                {eqDeltaClampWarning}
              </div>
            </div>
          )}

          {/* ฟอร์มตั้งค่า compressor แบบละเอียด จะแสดงเฉพาะเมื่อเลือก action นี้ */}
          {/* ===== ฟอร์มตั้งค่า Compressor แบบละเอียด ===== */}
          {action === "compressor" && (
            <div className="space-y-3">
              <div>
                <label className="block text-sm mb-1">Strength</label>
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

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs mb-1">Threshold (dBFS)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Preset"
                    value={compThreshold}
                    onChange={(e) => setCompThreshold(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Ratio</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Preset"
                    value={compRatio}
                    onChange={(e) => setCompRatio(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Attack (ms)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Preset"
                    value={compAttack}
                    onChange={(e) => setCompAttack(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Release (ms)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Preset"
                    value={compRelease}
                    onChange={(e) => setCompRelease(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Knee (dB)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={compKnee}
                    onChange={(e) => setCompKnee(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Makeup Gain (dB)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={compMakeupGain}
                    onChange={(e) => setCompMakeupGain(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Dry/Wet (%)</label>
                  <input
                    type="number"
                    step="1"
                    min="0"
                    max="100"
                    value={compDryWet}
                    onChange={(e) => setCompDryWet(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
                <div>
                  <label className="block text-xs mb-1">Output Ceiling (dBFS)</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="Off"
                    value={compOutputCeiling}
                    onChange={(e) => setCompOutputCeiling(e.target.value)}
                    className="w-full rounded-lg bg-[#0B1021] border border-[#5B21B6]/50 p-2 text-[#EDE9FE]"
                    disabled={loading}
                  />
                </div>
              </div>
            </div>
          )}

          {/* ฟอร์มปรับจำนวน half-steps สำหรับ pitch shift */}
          {/* ===== ฟอร์มตั้งค่าการเลื่อน pitch ===== */}
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

          {/* ปุ่มเริ่มประมวลผลจะส่งคำขอหลักไปยัง backend */}
          {/* ===== ปุ่มเริ่มประมวลผล ===== */}
          <button
            onClick={handleUpload}
            disabled={loading || (action === "eq-ai" && !isEqDeltaClampValid)}
            className={`w-full rounded-xl py-3 text-lg font-bold transition ${
              loading || (action === "eq-ai" && !isEqDeltaClampValid)
                ? "bg-[#A78BFA]/60 cursor-not-allowed"
                : "bg-[#5B21B6] hover:bg-[#22D3EE] text-white cursor-pointer"
            }`}
          >
            {loading ? "กำลังประมวลผล..." : "เริ่มประมวลผล"}
          </button>
          {/* แถบ progress และตัวเลขสถานะระหว่างรอผลจาก backend */}
          {/* ===== แถบ progress และข้อความสถานะ ===== */}
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
          {/* กลุ่มข้อความ feedback จากการประมวลผล เช่น status, เวลา, error และ success */}
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

      {/* คอลัมน์ขวา: ผลวิเคราะห์, ตัวเล่น stem, และลิงก์ดาวน์โหลดที่สร้างขึ้น */}
      {/* ส่วนแสดงผลวิเคราะห์ ตัวเล่น และลิงก์ดาวน์โหลด */}
      <div className="space-y-4">
        {/* การ์ดสรุป tempo / key / pitch ของไฟล์ต้นฉบับ */}
        {/* ===== การ์ดสรุป tempo / key / pitch ===== */}
        {analysis && <AudioAnalysis data={analysis} />}

        {/* ส่วนเครื่องเล่นหลายสเตม จะแสดงเมื่อ backend แยก stem สำเร็จแล้ว */}
        {/* ===== เครื่องเล่นแทร็กที่ถูกแยก stem แล้ว ===== */}
        {fileId && (
          <div className="rounded-2xl border border-[#5B21B6]/30 bg-[#0F172A] p-4 backdrop-blur">
            <h3 className="text-xl font-semibold mb-2">Multi-stem Player</h3>
            <MultiStemLivePlayer fileId={fileId} />
          </div>
        )}

        {/* ปุ่มดาวน์โหลด ZIP ของ stem ทั้งหมดจากงานแยกเสียง */}
        {/* ===== ปุ่มดาวน์โหลด ZIP ของ stem ทั้งหมด ===== */}
        {zipUrl && (
          <a
            href={zipUrl}
            download="separated.zip"
            className="block w-full text-center rounded-xl bg-[#22D3EE] hover:bg-[#5B21B6] text-black font-semibold py-3"
          >
            ดาวน์โหลดสเตมทั้งหมด (ZIP)
          </a>
        )}

        {/* ส่วนดาวน์โหลดไฟล์เดี่ยวที่ประมวลผลแล้ว พร้อม waveform player สำหรับฟังผลลัพธ์ */}
        {/* ===== ปุ่มดาวน์โหลดไฟล์ผลลัพธ์และ waveform player ===== */}
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
