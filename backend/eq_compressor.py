# backend/eq_compressor.py

import os
import torchaudio
import torch
from torchaudio.functional import equalizer_biquad
from pydub import AudioSegment

# EQ preset ต่อแนวเพลง (ปรับ gain/center_freq/Q ได้ตามต้องการ)
EQ_GENRE_PRESETS = {
    "pop": {"boost_mid": (1500, 4.0, 1.0), "air": (8000, 2.5, 0.8)},
    "rock": {"low_punch": (120, 3.5, 0.9), "presence": (3000, 3.0, 1.0)},
    "trap": {"sub": (60, 5.0, 1.2), "snap": (8000, 3.5, 0.9)},
    "country": {"body": (250, 2.5, 0.9), "clarity": (4000, 3.0, 0.9)},
    "soul": {"warm": (180, 3.0, 1.0), "silk": (6000, 2.5, 0.8)},
}

# Compressor preset ต่อแนวเพลง (ค่า default/generic)
COMP_GENRE_PRESETS = {
    "general": {"threshold": -24.0, "ratio": 4.0, "attack": 5, "release": 80},
    "pop": {"threshold": -22.0, "ratio": 3.5, "attack": 5, "release": 100},
    "rock": {"threshold": -26.0, "ratio": 5.0, "attack": 3, "release": 70},
    "trap": {"threshold": -28.0, "ratio": 6.0, "attack": 2, "release": 80},
    "country": {"threshold": -24.0, "ratio": 4.0, "attack": 8, "release": 120},
    "soul": {"threshold": -23.0, "ratio": 3.5, "attack": 6, "release": 110},
}


def _apply_genre_eq(waveform: torch.Tensor, sample_rate: int, genre: str) -> torch.Tensor:
    preset = EQ_GENRE_PRESETS.get(genre, {})
    for name, (freq, gain, q) in preset.items():
        waveform = equalizer_biquad(waveform, sample_rate, center_freq=freq, gain=gain, Q=q)
    return waveform


def apply_eq(waveform: torch.Tensor, sample_rate: int, genre: str) -> torch.Tensor:
    """
    Apply only genre EQ (ไม่มี preset ต่อ stem).
    """
    waveform = _apply_genre_eq(waveform, sample_rate, genre)
    return waveform


def apply_eq_to_file(input_path: str, genre: str, output_dir: str = "eq_applied") -> str:
    os.makedirs(output_dir, exist_ok=True)

    waveform, rate = torchaudio.load(input_path)
    eq_waveform = apply_eq(waveform, rate, genre)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base}_{genre}_eq.wav")
    torchaudio.save(output_path, eq_waveform.cpu(), sample_rate=rate)

    print(f"EQ (genre={genre}) เสร็จแล้ว: {output_path}")
    return output_path


def apply_compression(input_path: str, strength: str = "medium", genre: str = "general", output_dir: str = "compressed") -> str:
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{genre}_compressed.wav")

    audio = AudioSegment.from_wav(input_path)

    # เริ่มจาก preset ต่อ genre
    genre_kwargs = COMP_GENRE_PRESETS.get(genre, COMP_GENRE_PRESETS["general"]).copy()

    # ปรับตาม strength (แบบง่าย: scale threshold/ratio)
    if strength == "soft":
        genre_kwargs["threshold"] += 4.0
        genre_kwargs["ratio"] = max(2.0, genre_kwargs["ratio"] - 1.0)
    elif strength == "hard":
        genre_kwargs["threshold"] -= 4.0
        genre_kwargs["ratio"] = genre_kwargs["ratio"] + 1.5

    audio = audio.normalize().compress_dynamic_range(**genre_kwargs)
    audio.export(output_path, format="wav")
    print(f"Compression ({strength}, genre={genre}) เสร็จแล้ว: {output_path}")
    return output_path
