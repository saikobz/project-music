# backend/process_audio.py

import os
from openunmix.predict import separate
import torchaudio
import torch

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