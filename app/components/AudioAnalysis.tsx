"use client";
import React from "react";

interface AnalysisProps {
    data: {
        tempo: number;
        key: string;
        pitch: string | null;
    };
}

const AudioAnalysis: React.FC<AnalysisProps> = ({ data }) => {
    return (
        <div className="mt-4 p-4 bg-purple-700 rounded-lg text-white">
            <h3 className="font-semibold mb-2">🔍 Audio Analysis</h3>
            <div className="grid grid-cols-3 gap-2 text-center">
                <div>
                    <p className="text-sm">Tempo</p>
                    <p className="text-lg">{Math.round(data.tempo)} BPM</p>
                </div>
                <div>
                    <p className="text-sm">Key</p>
                    <p className="text-lg">{data.key}</p>
                </div>
                <div>
                    <p className="text-sm">Pitch</p>
                    <p className="text-lg">{data.pitch || '-'}</p>
                </div>
            </div>
        </div>
    );
};

export default AudioAnalysis;