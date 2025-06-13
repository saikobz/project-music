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

    const toggleMute = (stem: StemType) => {
        const ws = waveSurferRefs.current[stem];
        const isMuted = !mutedTracks[stem];
        if (ws) {
            ws.setVolume(isMuted ? 0 : 1);
            setMutedTracks((prev) => ({ ...prev, [stem]: isMuted }));
        }
    };

    return (
        <div className="space-y-6">
            <button
                onClick={togglePlay}
                className="px-4 py-2 bg-green-600 text-white rounded"
            >
                {isPlaying ? "⏸ หยุดทั้งหมด" : "▶️ เล่นทั้งหมด"}
            </button>

            {stems.map((stem) => (
                <div key={stem} className="space-y-2">
                    <div className="flex justify-between items-center">
                        <span className="capitalize font-bold">{stem}</span>
                        <button
                            onClick={() => toggleMute(stem)}
                            className="text-sm px-3 py-1 border rounded"
                        >
                            {mutedTracks[stem] ? "Unmute" : "Mute"}
                        </button>
                    </div>

                    <div id={`waveform-${stem}`} className="bg-gray-200 rounded" />
                </div>
            ))}
        </div>
    );
}
