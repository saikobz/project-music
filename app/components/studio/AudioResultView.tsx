// app/components/studio/AudioResultView.tsx
"use client";

import React from "react";
import WaveformPlayer from "../WaveformPlayer";
import MultiStemLivePlayer from "../MultiStemLivePlayer";
import AudioAnalysis from "../AudioAnalysis";
import ExportMasterModal from "../ExportMasterModal";
import SingleExportModal from "../SingleExportModal";

// Skeletons
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

interface AudioResultViewProps {
  loading: boolean;
  action: string;
  analysis: { tempo: number; key: string; pitch: string | null } | null;
  fileId: string | null;
  zipUrl: string | null;
  downloadUrl: string | null;
  downloadFileName: string | null;
  exportFormat: string;
  apiBase: string;
  isExportModalOpen: boolean;
  setIsExportModalOpen: (open: boolean) => void;
  isExporting: boolean;
  handleExport: (type: string, format: string, lufs: number, stems: string[]) => Promise<void>;
  isSingleExportModalOpen: boolean;
  setIsSingleExportModalOpen: (open: boolean) => void;
  handleSingleExport: (format: string) => void;
}

export function AudioResultView({
  loading,
  action,
  analysis,
  fileId,
  zipUrl,
  downloadUrl,
  downloadFileName,
  exportFormat,
  apiBase,
  isExportModalOpen,
  setIsExportModalOpen,
  isExporting,
  handleExport,
  isSingleExportModalOpen,
  setIsSingleExportModalOpen,
  handleSingleExport,
}: AudioResultViewProps) {
  return (
    <div className="space-y-4">
      {loading && (
        <>
          <AudioAnalysisSkeleton />
          {action === "separate" ? <StemMixerSkeleton /> : <SinglePlayerSkeleton />}
        </>
      )}

      {!loading && (analysis || fileId || downloadUrl) && (
        <div className="animate-fade-in-up space-y-4">
          {analysis && <AudioAnalysis data={analysis} />}

          {fileId && action === "separate" && (
            <div>
              <MultiStemLivePlayer fileId={fileId} />
            </div>
          )}

          {zipUrl && action === "separate" && (
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
                  href={`${apiBase}/karaoke/${fileId}?export_format=${exportFormat}`}
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

          {downloadUrl && downloadFileName && !downloadFileName.endsWith(".zip") && (
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
    </div>
  );
}
