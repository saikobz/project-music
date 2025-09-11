"use client";
import React, { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";

const stems = ["vocals", "drums", "bass", "other"] as const;
type StemType = typeof stems[number];

type Props = {
    baseUrl: string; // ex: http://localhost:8000/separated/<fileId>
};

export default function AdvancedMultiTrackPlayer({ baseUrl }: Props) {
    const waveSurferRefs = useRef<Record<StemType, WaveSurfer | null>>({
        vocals: null,
        drums: null,
        bass: null,
        other: null,
    });

    const [isPlaying, setIsPlaying] = useState(false);
    const [mutedTracks, setMutedTracks] = useState<Record<StemType, boolean>>({
        vocals: false,
        drums: false,
        bass: false,
        other: false,
    });

    useEffect(() => {
        stems.forEach((stem) => {
            const container = document.getElementById(`waveform-${stem}`);
            if (!container) return;

            // ✅ ล้าง container ก่อน (กันคลื่นซ้อน)
            container.innerHTML = "";

            // ✅ destroy ตัวเก่าถ้ามี
            waveSurferRefs.current[stem]?.destroy();

            const ws = WaveSurfer.create({
                container,
                waveColor: "#ccc",
                progressColor: "#10b981",
                height: 60,
            });

            ws.load(`${baseUrl}/${stem}.wav`);

            ws.on("ready", () => {
                waveSurferRefs.current[stem] = ws;
                if (mutedTracks[stem]) ws.setVolume(0);
            });
        });

        return () => {
            stems.forEach((stem) => {
                waveSurferRefs.current[stem]?.destroy();
            });
        };
    }, [baseUrl]);

    const togglePlay = () => {
        stems.forEach((stem) => {
            const ws = waveSurferRefs.current[stem];
            if (ws) ws.playPause();
        });
        setIsPlaying((prev) => !prev);
    };

    const resetAll = () => {
        stems.forEach((stem) => {
            const ws = waveSurferRefs.current[stem];
            if (ws) {
                ws.pause();
                ws.seekTo(0);
            }
        });
        setIsPlaying(false);
    };

    const toggleMute = (stem: StemType) => {
        const ws = waveSurferRefs.current[stem];
        const isMuted = !mutedTracks[stem];
        if (ws) {
            ws.setVolume(isMuted ? 0 : 1);
            setMutedTracks((prev) => ({ ...prev, [stem]: isMuted }));
        }
    };

    const handleDownload = async (stem: StemType) => {
        try {
            const response = await fetch(`${baseUrl}/${stem}.wav`);
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `${stem}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Download failed", error);
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex gap-2">
                <button
                    onClick={togglePlay}
                    className="px-4 py-2 bg-green-600 text-white rounded"
                >
                    {isPlaying ? "⏸ หยุดทั้งหมด" : "▶️ เล่นทั้งหมด"}
                </button>
                <button
                    onClick={resetAll}
                    className="px-4 py-2 bg-gray-500 text-white rounded"
                >
                    ⏮️ รีเซ็ต
                </button>
            </div>

            {stems.map((stem) => (
                <div key={stem} className="space-y-2">
                    <div className="flex justify-between items-center">
                        <span className="capitalize font-bold">{stem}</span>
                        <div className="flex gap-2">
                            <button
                                onClick={() => handleDownload(stem)}
                                className="text-sm px-3 py-1 border rounded bg-white text-black"
                            >
                                ดาวน์โหลด
                            </button>
                            <button
                                onClick={() => toggleMute(stem)}
                                className="text-sm px-3 py-1 border rounded"
                            >
                                {mutedTracks[stem] ? "Unmute" : "Mute"}
                            </button>
                        </div>
                    </div>

                    <div id={`waveform-${stem}`} className="bg-gray-200 rounded" />
                </div>
            ))}
        </div>
    );
}
