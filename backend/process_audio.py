# backend/process_audio.py

import os
from openunmix.predict import separate
import torchaudio
import torch
import numpy as np
import librosa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def separate_audio(input_path: str, output_dir: str = "separated") -> str:
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[-1].lower()
    if ext != ".wav":
        raise ValueError("Only .wav files are supported.")

    try:
        # โหลดไฟล์เสียง
        audio_tensor, rate = torchaudio.load(input_path)

        # แยกเสียงด้วย Open-Unmix
        estimates = separate(
            audio=audio_tensor.to(DEVICE),
            rate=rate,
            targets=["vocals", "drums", "bass", "other"],
            device=str(DEVICE)
        )

        # บันทึกไฟล์ที่แยกได้ (ไม่มี EQ)
        for target, waveform in estimates.items():
            if waveform.ndim == 3:
                waveform = waveform.squeeze(0)

            torchaudio.save(
                os.path.join(output_dir, f"{target}.wav"),
                waveform.cpu(),  # สำคัญมาก: ต้อง .cpu() ก่อน save
                sample_rate=rate
            )

        print("✅ แยกเสียงเสร็จเรียบร้อยแล้ว!")
        print("📁 ไฟล์อยู่ที่:", output_dir)
        return output_dir

    except Exception as e:
        print(f"❌ ERROR in separate_audio: {e}")
        raise


def analyze_audio(input_path: str) -> dict:
    """Analyze audio file and return tempo, pitch, and key information."""
    try:
        y, sr = librosa.load(input_path, sr=None, mono=True)

        # Tempo estimation
        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])

        # Pitch estimation using YIN
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
        f0 = f0[~np.isnan(f0)]
        pitch_note = None
        if f0.size:
            pitch_hz = float(np.median(f0))
            pitch_note = librosa.hz_to_note(pitch_hz)

        # Key estimation using chroma features and major/minor profiles
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        maj_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        maj_scores = [np.correlate(chroma_mean, np.roll(maj_profile, i))[0] for i in range(12)]
        min_scores = [np.correlate(chroma_mean, np.roll(min_profile, i))[0] for i in range(12)]

        maj_idx = int(np.argmax(maj_scores))
        min_idx = int(np.argmax(min_scores))

        if maj_scores[maj_idx] >= min_scores[min_idx]:
            key = f"{keys[maj_idx]} major"
        else:
            key = f"{keys[min_idx]} minor"

        return {"tempo": tempo, "pitch": pitch_note, "key": key}
    except Exception as e:
        print(f"❌ ERROR in analyze_audio: {e}")
        raise