"use client";
import React from "react";
import AdvancedMultiTrackPlayer from "./AdvancedMultiTrackPlayer";
import { API_BASE_URL } from "@/lib/config";

type Props = {
    fileId: string;
};

// ตัวกลางที่แปลง file id จาก backend ให้เป็น base URL ที่ตัวเล่นหลายสเตมใช้งานต่อได้
export default function MultiStemLivePlayer({ fileId }: Props) {
    const baseUrl = `${API_BASE_URL}/separated/${fileId}`;
    return <AdvancedMultiTrackPlayer baseUrl={baseUrl} fileId={fileId} />;
}
