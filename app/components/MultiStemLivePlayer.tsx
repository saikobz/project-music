"use client";
import React from "react";
import AdvancedMultiTrackPlayer from "./AdvancedMultiTrackPlayer";

type Props = {
    fileId: string;
};

// ตัวกลางที่แปลง file id จาก backend ให้เป็น base URL ที่ตัวเล่นหลายสเตมใช้งานต่อได้
export default function MultiStemLivePlayer({ fileId }: Props) {
    const baseUrl = `http://localhost:8000/separated/${fileId}`;
    return <AdvancedMultiTrackPlayer baseUrl={baseUrl} />;
}
