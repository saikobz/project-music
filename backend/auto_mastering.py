import logging
import os
import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, HighShelfFilter, Limiter
import pyloudnorm as pyln

logger = logging.getLogger(__name__)

def polish_vocal_file(input_path: str, output_path: str) -> None:
    """ขัดเกลาเสียงร้องด้วย De-esser, Compressor และ Air EQ"""
    data, sr = sf.read(input_path)
    board = Pedalboard([
        Compressor(threshold_db=-15, ratio=3.0, attack_ms=5.0, release_ms=50.0),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=2.0)
    ])
    processed = board(data, sr)
    sf.write(output_path, processed, sr)

def apply_lufs_mastering(input_path: str, output_path: str, target_lufs: float) -> None:
    """ปรับความดัง LUFS และใส่ True Peak Limiter"""
    data, sr = sf.read(input_path)

    # pyloudnorm ต้องการ 2D array เสมอ (samples, channels)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)

    # วัด LUFS ปัจจุบัน
    meter = pyln.Meter(sr)
    try:
        current_lufs = meter.integrated_loudness(data)
    except Exception as exc:
        logger.warning(f"ไม่สามารถวัด LUFS ได้: {exc}")
        current_lufs = np.nan

    if np.isfinite(current_lufs):
        # คำนวณส่วนต่าง Gain
        delta_lufs = target_lufs - current_lufs
        gain_linear = 10.0 ** (delta_lufs / 20.0)
        audio_gain = data * gain_linear
    else:
        # ไฟล์เงียบสนิทหรือสั้นเกินไป: integrated_loudness คืน -inf/NaN
        # ข้ามขั้นตอนปรับ gain เพื่อไม่ให้เกิด NaN (0 * inf) แล้วเขียน garbage ลงไฟล์
        logger.warning("ไฟล์เงียบหรือสั้นเกินไป ข้ามขั้นตอนปรับความดัง (ใช้ Limiter อย่างเดียว)")
        audio_gain = data

    # ป้องกันเสียงแตก (Clipping) ด้วย Limiter ที่ -1.0 dB
    board = Pedalboard([
        Limiter(threshold_db=-1.0)
    ])

    mastered = board(audio_gain, sr)
    sf.write(output_path, mastered, sr)
