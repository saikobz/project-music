# backend/eq_compressor.py

import os
import torchaudio
import torch
from torchaudio.functional import equalizer_biquad
from pydub import AudioSegment

def apply_eq(waveform: torch.Tensor, sample_rate: int, target: str) -> torch.Tensor:
    """
    ปรับ EQ ตามประเภทเช่น vocals, drums, ...
    """
    if target == "vocals":
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=1500, gain=6.0, Q=1.0)
    elif target == "drums":
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=100, gain=-4.0, Q=0.7)
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=8000, gain=5.0, Q=0.7)
    elif target == "bass":
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=80, gain=6.0, Q=0.8)
    elif target == "other":
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=4000, gain=3.0, Q=0.5)
    else:
        print(f"⚠️ ไม่รู้จัก target: {target}, ข้าม EQ ไปเลย")

    return waveform

def apply_eq_to_file(input_path: str, target: str, output_dir: str = "eq_applied") -> str:
    os.makedirs(output_dir, exist_ok=True)

    waveform, rate = torchaudio.load(input_path)
    eq_waveform = apply_eq(waveform, rate, target)

    output_path = os.path.join(output_dir, f"{target}_eq.wav")
    torchaudio.save(output_path, eq_waveform.cpu(), sample_rate=rate)

    print(f"✅ EQ สำหรับ {target} เสร็จแล้ว: {output_path}")
    return output_path

def apply_compression(input_path: str, strength: str = "medium", output_dir: str = "compressed") -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_compressed.wav")

    audio = AudioSegment.from_wav(input_path)

    if strength == "soft":
        kwargs = dict(threshold=-20.0, ratio=2.0)
    elif strength == "hard":
        kwargs = dict(threshold=-30.0, ratio=6.0)
    else:
        kwargs = dict(threshold=-24.0, ratio=4.0)

    audio = audio.normalize().compress_dynamic_range(**kwargs)
    audio.export(output_path, format="wav")
    print(f"✅ Compression ({strength}) เสร็จแล้ว: {output_path}")
    return output_path