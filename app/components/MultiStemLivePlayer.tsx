"use client";
import React from "react";
import AdvancedMultiTrackPlayer from "./AdvancedMultiTrackPlayer";

type Props = {
    fileId: string;
};

export default function MultiStemLivePlayer({ fileId }: Props) {
    const baseUrl = `http://localhost:8000/separated/${fileId}`;
    return <AdvancedMultiTrackPlayer baseUrl={baseUrl} />;
}
