"use client";
import React, { useState } from 'react';
import axios from 'axios';
import WaveformPlayer from "./WaveformPlayer";
import MultiStemLivePlayer from "./MultiStemLivePlayer";
import AudioAnalysis from "./AudioAnalysis";

function UploadBox() {
    const [file, setFile] = useState<File | null>(null);
    const [action, setAction] = useState('separate');
    const [target, setTarget] = useState('vocals');
    const [strength, setStrength] = useState('medium');
    const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
    const [downloadFileName, setDownloadFileName] = useState<string | null>(null);
    const [processingTime, setProcessingTime] = useState<string | null>(null);
    const [fileId, setFileId] = useState<string | null>(null); // ✅ เก็บ file_id จาก backend
    const [zipUrl, setZipUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [analysis, setAnalysis] = useState<{ tempo: number; key: string; pitch: string | null } | null>(null);

    const handleUpload = async () => {
        if (!file) return alert('กรุณาเลือกไฟล์ก่อน');
        setLoading(true);
        setDownloadUrl(null);
        setDownloadFileName(null);
        setProcessingTime(null);
        setFileId(null);
        setZipUrl(null);
        setErrorMessage(null);
        setAnalysis(null);

        const formData = new FormData();
        formData.append('file', file);

        const startTime = Date.now();

        try {
            let response;
            let suffix = '';

            if (action === 'separate') {
                response = await axios.post('http://localhost:8000/separate', formData);
                const { file_id, zip_url } = response.data;
                setFileId(file_id);
                setZipUrl(zip_url);
                alert('✅ แยกเสียงเสร็จแล้ว');
            }

            if (action === 'eq') {
                response = await axios.post(`http://localhost:8000/apply-eq?target=${target}`, formData, {
                    responseType: 'blob'
                });
                const url = window.URL.createObjectURL(new Blob([response.data]));
                setDownloadUrl(url);
                suffix = `_eq_${target}`;
            }

            if (action === 'compressor') {
                response = await axios.post(`http://localhost:8000/apply-compressor?strength=${strength}`, formData, {
                    responseType: 'blob'
                });
                const url = window.URL.createObjectURL(new Blob([response.data]));
                setDownloadUrl(url);
                suffix = `_compressed_${strength}`;
            }

            if (file && suffix) {
                const baseName = file.name.replace(/\.[^/.]+$/, '');
                setDownloadFileName(`${baseName}${suffix}.wav`);
            }

            const analyzeData = new FormData();
            analyzeData.append('file', file);
            try {
                const analyzeResp = await axios.post('http://localhost:8000/analyze', analyzeData);
                setAnalysis(analyzeResp.data);
            } catch (err) {
                console.error('Analyze error', err);
            }

            const endTime = Date.now();
            const duration = Math.floor((endTime - startTime) / 1000);
            const minutes = Math.floor(duration / 60);
            const seconds = duration % 60;
            setProcessingTime(`${minutes} นาที ${seconds} วินาที`);
        } catch (err: any) {
            let message = '❌ เกิดข้อผิดพลาด';
            if (err.code === 'ERR_NETWORK') {
                message = '🚫 ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ (อาจเกิดจาก CORS หรือเซิร์ฟเวอร์ล้ม)';
            } else if (err.response?.status) {
                const status = err.response.status;
                if (status >= 500) {
                    message = `⚠️ เซิร์ฟเวอร์มีปัญหา (${status})`;
                } else {
                    message = `⚠️ คำขอถูกปฏิเสธ (${status})`;
                }
            }
            setErrorMessage(message);
            alert(message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 bg-purple-600 text-white rounded-xl shadow-xl max-w-xl mx-auto mt-8 space-y-4">
            <h2 className="text-center text-2xl font-bold">🎧 อัปโหลดไฟล์เสียง</h2>

            <input type="file" accept="audio/wav" onChange={(e) => setFile(e.target.files?.[0] || null)} className="text-black w-full p-2 bg-amber-50 rounded-lg" />

            <div>
                <label className="block mt-2">เลือกฟังก์ชัน:</label>
                <select value={action} onChange={(e) => setAction(e.target.value)} className="w-full text-black p-2 bg-amber-50 rounded-lg">
                    <option value="separate">แยกเสียง (AI Stem)</option>
                    <option value="eq">ปรับ EQ</option>
                    <option value="compressor">ปรับ Compressor</option>
                </select>
            </div>

            {action === 'eq' && (
                <div>
                    <label className="block mt-2">Preset EQ:</label>
                    <select value={target} onChange={(e) => setTarget(e.target.value)} className="w-full text-black p-2 rounded-lg bg-amber-50">
                        <option value="vocals">Vocals</option>
                        <option value="drums">Drums</option>
                        <option value="bass">Bass</option>
                        <option value="other">Other</option>
                    </select>
                </div>
            )}

            {action === 'compressor' && (
                <div>
                    <label className="block mt-2">ระดับความแรง (Strength):</label>
                    <select value={strength} onChange={(e) => setStrength(e.target.value)} className="w-full text-black p-2 rounded-lg bg-amber-50">
                        <option value="soft">Soft</option>
                        <option value="medium">Medium</option>
                        <option value="hard">Hard</option>
                    </select>
                </div>
            )}

            <button
                onClick={handleUpload}
                disabled={loading}
                className="mt-4 w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg cursor-pointer"
            >
                {loading ? 'กำลังประมวลผล...' : 'อัปโหลดและเริ่มทำงาน'}
            </button>

            {processingTime && (
                <div className="text-center text-green-300 font-medium mt-2">
                    ⏱️ เวลาที่ใช้ในการประมวลผล: {processingTime}
                </div>
            )}

            {errorMessage && (
                <div className="text-center text-red-300 font-medium mt-2">
                    {errorMessage}
                </div>
            )}

            {analysis && (
                <AudioAnalysis data={analysis} />
            )}
            
            {/* ✅ แสดง Player แบบเต็มหลังแยกเสียง */}
            {fileId && (
                <div className="mt-6">
                    <MultiStemLivePlayer fileId={fileId} />
                </div>
            )}

            {/* ✅ ลิงก์ดาวน์โหลด zip หากมี */}
            {zipUrl && (
                <a
                    href={zipUrl}
                    download="separated.zip"
                    className="mt-4 block w-full text-center bg-yellow-600 hover:bg-yellow-700 text-white font-semibold py-2 px-4 rounded-lg"
                >
                    📦 ดาวน์โหลดไฟล์ทั้งหมด (ZIP)
                </a>
            )}

            {/* 🎧 แสดง player ถ้าเป็น .wav เดี่ยว */}
            {downloadUrl && downloadFileName && !downloadFileName.endsWith('.zip') && (
                <WaveformPlayer audioUrl={downloadUrl} />
            )}
        </div>
    );
}

export default UploadBox;
