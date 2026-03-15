"""
หน้าที่ของไฟล์นี้:
- เก็บ logic การบีบอัดเสียง (compressor) ที่ใช้ทั้ง preset ตาม genre และค่าที่ผู้ใช้ override เอง
- คำนวณ envelope, gain reduction, dry/wet mix และ output ceiling ในระดับสัญญาณเสียงจริง
- ส่งออกไฟล์ WAV ที่ผ่านการ compression แล้วกลับไปให้ API layer ใช้งานต่อ
"""

import math
import os

import numpy as np
import torchaudio
import torch

# preset แต่ละ genre ใช้เป็นค่าเริ่มต้นก่อนจะถูกปรับเพิ่มด้วย strength หรือ override จากผู้ใช้
COMP_GENRE_PRESETS = {
    "general": {"threshold": -24.0, "ratio": 4.0, "attack": 5, "release": 80},
    "pop": {"threshold": -22.0, "ratio": 3.5, "attack": 5, "release": 100},
    "rock": {"threshold": -26.0, "ratio": 5.0, "attack": 3, "release": 70},
    "trap": {"threshold": -28.0, "ratio": 6.0, "attack": 2, "release": 80},
    "country": {"threshold": -24.0, "ratio": 4.0, "attack": 8, "release": 120},
    "soul": {"threshold": -23.0, "ratio": 3.5, "attack": 6, "release": 110},
}


def _normalize_mix_value(dry_wet: float) -> float:
    # รองรับทั้งรูปแบบ 0..1 และ 0..100 เพื่อให้เรียกใช้จาก API ได้สะดวก
    mix = float(dry_wet)
    if mix < 0.0:
        raise ValueError("dry_wet must be >= 0.")
    if mix > 1.0:
        # ยอมรับทั้งรูปแบบ 0..1 และ 0..100 เพื่อให้เรียกจาก API ได้สะดวกขึ้น
        if mix <= 100.0:
            mix /= 100.0
        else:
            raise ValueError("dry_wet must be in range 0..1 or 0..100.")
    return mix


def _compute_gain_reduction_db(
    levels_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray:
    # แปลง threshold / ratio / knee ให้เป็นเส้นโค้งการลด gain ในหน่วย dB
    slope = 1.0 - (1.0 / ratio)
    if knee_db <= 0.0:
        over = np.maximum(levels_db - threshold_db, 0.0)
        return -(over * slope)

    # soft knee จะค่อย ๆ เริ่มลด gain รอบ threshold แทนที่จะหักมุมทันที
    lower = threshold_db - (knee_db / 2.0)
    upper = threshold_db + (knee_db / 2.0)
    reduction_db = np.zeros_like(levels_db, dtype=np.float32)

    high = levels_db >= upper
    if np.any(high):
        over = levels_db[high] - threshold_db
        reduction_db[high] = -(over * slope)

    mid = (levels_db > lower) & (levels_db < upper)
    if np.any(mid):
        x = levels_db[mid] - lower
        reduction_db[mid] = -(slope * (x**2) / (2.0 * knee_db))

    return reduction_db


def _compress_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
    ratio: float,
    attack: float,
    release: float,
    knee: float,
    makeup_gain: float,
    dry_wet: float,
    output_ceiling: float | None,
    control_hop: int = 64,
) -> torch.Tensor:
    # แกนหลักของ compressor:
    # สร้าง envelope, คำนวณ gain reduction, แล้วผสม dry/wet ก่อนคืนผลลัพธ์
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0.")
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1.")
    if attack <= 0.0:
        raise ValueError("attack must be > 0.")
    if release <= 0.0:
        raise ValueError("release must be > 0.")
    if knee < 0.0:
        raise ValueError("knee must be >= 0.")
    if output_ceiling is not None and output_ceiling > 0.0:
        raise ValueError("output_ceiling must be <= 0 dBFS.")

    # แปลง Tensor เป็น numpy เพื่อคำนวณระดับสัญญาณแบบ sample-by-sample ได้ง่าย
    mix = _normalize_mix_value(dry_wet)

    wave = waveform.detach().cpu().numpy().astype(np.float32)
    if wave.ndim == 1:
        wave = wave[np.newaxis, :]
    if wave.shape[1] == 0:
        return waveform

    dry = wave.copy()
    # ใช้ peak ของทุก channel เป็น sidechain กลางสำหรับบีบอัดทั้งสัญญาณ
    sidechain = np.max(np.abs(wave), axis=0)

    hop = max(int(control_hop), 1)
    pad_len = (-sidechain.shape[0]) % hop
    if pad_len > 0:
        sidechain_padded = np.pad(sidechain, (0, pad_len), mode="constant", constant_values=0.0)
    else:
        sidechain_padded = sidechain

    peaks = sidechain_padded.reshape(-1, hop).max(axis=1).astype(np.float32)
    envelope = np.zeros_like(peaks, dtype=np.float32)

    # attack/release กำหนดความเร็วในการตอบสนองของ envelope เมื่อระดับสัญญาณเปลี่ยน
    attack_s = max(float(attack), 0.1) / 1000.0
    release_s = max(float(release), 0.1) / 1000.0
    frame_time_s = hop / float(sample_rate)
    attack_coeff = math.exp(-frame_time_s / attack_s)
    release_coeff = math.exp(-frame_time_s / release_s)

    env_prev = 0.0
    for idx, peak in enumerate(peaks):
        # ใช้ attack/release คนละ coefficient เพื่อให้ envelope ตอบสนองแบบ compressor จริง
        coeff = attack_coeff if peak > env_prev else release_coeff
        env_prev = coeff * env_prev + (1.0 - coeff) * peak
        envelope[idx] = env_prev

    levels_db = 20.0 * np.log10(np.maximum(envelope, 1e-8))
    reduction_db = _compute_gain_reduction_db(
        levels_db,
        threshold_db=float(threshold),
        ratio=float(ratio),
        knee_db=float(knee),
    )

    total_gain_db = reduction_db + float(makeup_gain)
    gain_frames = np.power(10.0, total_gain_db / 20.0).astype(np.float32)
    # ขยาย gain จากระดับ frame control ให้กลับมาเป็น sample-by-sample ก่อนคูณกับคลื่นเสียง
    gain_samples = np.repeat(gain_frames, hop)[: sidechain.shape[0]]

    wet = wave * gain_samples[np.newaxis, :]
    # dry/wet ช่วยผสมสัญญาณเดิมกลับเข้าไปได้แบบ parallel compression
    mixed = dry * (1.0 - mix) + wet * mix

    if output_ceiling is not None:
        # จำกัด peak ขั้นสุดท้ายเพื่อไม่ให้ดังเกิน ceiling ที่ผู้ใช้ตั้งไว้
        ceiling_lin = 10.0 ** (float(output_ceiling) / 20.0)
        peak = float(np.max(np.abs(mixed)))
        if peak > ceiling_lin and peak > 1e-8:
            mixed *= ceiling_lin / peak

    mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
    return torch.from_numpy(mixed)


def apply_compression(
    input_path: str,
    strength: str = "medium",
    genre: str = "general",
    output_dir: str = "compressed",
    *,
    threshold: float | None = None,
    ratio: float | None = None,
    attack: float | None = None,
    release: float | None = None,
    knee: float | None = None,
    makeup_gain: float = 0.0,
    dry_wet: float = 100.0,
    output_ceiling: float | None = None,
) -> str:
    # ฟังก์ชันนี้เป็นตัวเชื่อมระหว่าง API layer กับ DSP จริง:
    # โหลดไฟล์, รวม preset + override, บีบอัดเสียง, แล้วบันทึกไฟล์ผลลัพธ์
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_{genre}_compressed.wav")

    waveform, sample_rate = torchaudio.load(input_path)

    # เริ่มจาก preset ตาม genre แล้วค่อยขยับตาม strength และค่าที่ผู้ใช้ส่ง override มา
    genre_kwargs = COMP_GENRE_PRESETS.get(genre, COMP_GENRE_PRESETS["general"]).copy()

    if strength == "soft":
        genre_kwargs["threshold"] += 4.0
        genre_kwargs["ratio"] = max(2.0, genre_kwargs["ratio"] - 1.0)
    elif strength == "hard":
        genre_kwargs["threshold"] -= 4.0
        genre_kwargs["ratio"] = genre_kwargs["ratio"] + 1.5
    elif strength != "medium":
        raise ValueError(f"Invalid strength '{strength}'. Use soft, medium, or hard.")

    if threshold is not None:
        genre_kwargs["threshold"] = float(threshold)
    if ratio is not None:
        genre_kwargs["ratio"] = float(ratio)
    if attack is not None:
        genre_kwargs["attack"] = float(attack)
    if release is not None:
        genre_kwargs["release"] = float(release)
    knee_value = 6.0 if knee is None else float(knee)

    # ส่งค่าทั้งหมดเข้าแกน compressor แล้วเซฟผลเป็นไฟล์ WAV
    compressed = _compress_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        threshold=genre_kwargs["threshold"],
        ratio=genre_kwargs["ratio"],
        attack=genre_kwargs["attack"],
        release=genre_kwargs["release"],
        knee=knee_value,
        makeup_gain=float(makeup_gain),
        dry_wet=float(dry_wet),
        output_ceiling=output_ceiling,
    )

    torchaudio.save(output_path, compressed.cpu(), sample_rate=sample_rate)
    print(f"Compression ({strength}, genre={genre}) done: {output_path}")
    return output_path
