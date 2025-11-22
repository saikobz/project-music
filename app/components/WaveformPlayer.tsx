// frontend/components/WaveformPlayer.tsx
"use client";
import React, { useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";

interface WaveformPlayerProps {
    audioUrl: string;
}

const WaveformPlayer: React.FC<WaveformPlayerProps> = ({ audioUrl }) => {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const waveSurferRef = useRef<WaveSurfer | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        if (waveSurferRef.current) {
            waveSurferRef.current.destroy();
        }

        waveSurferRef.current = WaveSurfer.create({
            container: containerRef.current,
            waveColor: "#ddd",
            progressColor: "#facc15",
            height: 80,
        });

        waveSurferRef.current.load(audioUrl);

        return () => waveSurferRef.current?.destroy();
    }, [audioUrl]);

    return (
        <div className="mt-4 space-y-2">
            <div ref={containerRef} />
            <div className="text-center">
                <button
                    onClick={() => waveSurferRef.current?.playPause()}
                    className="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-1 px-4 rounded-lg cursor-pointer"
                >
                    ▶️ เล่น / ⏸️ หยุด
                </button>
            </div>
        </div>
    );
};

export default WaveformPlayer;
