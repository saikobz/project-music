# -*- coding: utf-8 -*-
"""
สำหรับรัน Auto-EQ inference 
ตัว “ประมวลผลเสียงด้วยโมเดล Auto-EQ” หลังจากโมเดลถูก train มาแล้ว 
โดยหน้าที่หลักไม่ใช่การสอนโมเดล แต่เป็นการเอาไฟล์เสียงหรือ waveform เข้ามา 
แล้วให้โมเดลทำนายว่าโทนเสียงควรถูกปรับ EQ ไปทางไหน จากนั้นแปลงผลทำนายนั้นกลับไปเป็นเสียงที่ถูกปรับแล้ว
"""

import logging
import os
from functools import lru_cache
from typing import Any, Mapping

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

# sample rate 
SR = 44100
# ขนาด FFT สำหรับสร้างและแปลง spectrogram
N_FFT = 2048
# จำนวน sample ที่เลื่อนในแต่ละเฟรมของ STFT
HOP = 512
# จำนวน mel bins ของ mel spectrogram
N_MELS = 128
# ช่วง dB สูงสุดที่ใช้ normalize และ denormalize mel
TOP_DB = 80.0
# ความยาวต่อ chunk สำหรับประมวลผลทีละช่วง
SEGMENT_SECONDS = 5
# ระยะ overlap ระหว่าง chunk เพื่อใช้ crossfade
OVERLAP_SECONDS = 0.50
# ค่าต่ำสุดของ mel ในหน่วย dB
MEL_DB_MIN = -80.0
# ค่าสูงสุดของ mel ในหน่วย dB
MEL_DB_MAX = 0.0
# เพดานการปรับ EQ ต่อจุดเพื่อกันการเปลี่ยนแรงเกินไป
DELTA_CLAMP_DB = 2.0
# จำนวน tap สำหรับ smooth ตามแกนความถี่
DELTA_SMOOTH_FREQ_BINS = 7
# จำนวน tap สำหรับ smooth ตามแกนเวลา
DELTA_SMOOTH_TIME_FRAMES = 7
# สัดส่วนการ blend เสียงที่ผ่าน Auto-EQ กลับเข้ากับเสียงเดิม
OUTPUT_BLEND = 0.20
# เพดานการชดเชยความดังแบบ RMS
MAX_RMS_GAIN_DB = 1.0
# ค่าสูงสุดของ peak หลังประมวลผลเพื่อกัน clipping
PEAK_LIMIT = 0.98
# จุดเริ่มลดแรง EQ ในย่านความถี่สูง
HF_TAPER_START = 0.65
# ค่าปลายของแรง EQ หลัง taper ย่านความถี่สูง
HF_TAPER_END = 0.35
# ค่าต่ำสุดของ adaptive blend scale
MIN_BLEND_SCALE = 0.45
# ค่าต่ำสุดของ adaptive delta scale
MIN_DELTA_SCALE = 0.50
# เกณฑ์ตรวจช่วงเสียงเบาที่ควรลดแรงประมวลผล
LOW_RMS_THRESHOLD = 0.035
# เกณฑ์ตรวจว่าสัดส่วนพลังงานย่านสูงมากเกินไปหรือไม่
HIGH_HF_RATIO_THRESHOLD = 0.22

# รายชื่อ genre ที่ checkpoint รุ่น FiLM รองรับ
GENRES = ["trap", "pop", "rock", "soul", "country"]
# map ชื่อ genre ไปเป็นเลข id เพื่อใช้ป้อนเข้า embedding layer
GENRE2ID = {genre: idx for idx, genre in enumerate(GENRES)}
# genre เริ่มต้นเมื่อ caller ไม่ส่งค่าเข้ามา
DEFAULT_GENRE = "pop"

# path ของ checkpoint หลักที่ใช้โหลดโมเดล Auto-EQ
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoeq_cnn_v1.pt")
# logger ของโมดูลนี้ ใช้บันทึกเหตุการณ์โหลดโมเดลและ fallback ต่าง ๆ
logger = logging.getLogger(__name__)


class AutoEQModelLoadError(RuntimeError):
    """ใช้เมื่อไม่สามารถโหลด checkpoint ของ Auto-EQ ได้"""
class AutoEQ(nn.Module):
    # โมเดลรุ่นปัจจุบันที่ใช้ genre embedding มาปรับ feature map ผ่าน FiLM
    def __init__(self, n_genres: int, ch: int = 256):
        super().__init__()
        # embedding แปลง genre id เป็นเวกเตอร์เงื่อนไขสำหรับควบคุมโมเดล
        self.emb = nn.Embedding(n_genres, 32)

        # stem ทำหน้าที่ดึง feature แรกเริ่มจาก mel spectrogram เข้า latent space
        self.stem = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),  # convolution ชั้นแรก รับ input 1 channel แล้วแปลงเป็น feature map จำนวน ch channel
            nn.GELU(),  # activation function ช่วยให้โมเดลเรียนรู้ความสัมพันธ์ที่ไม่เป็นเส้นตรง
            nn.Conv2d(ch, ch, 3, padding=1),  # convolution ชั้นที่ 2 ใช้สกัด feature ต่อโดยคงจำนวน channel และขนาดเดิม
            nn.GELU(),  # activation function หลัง convolution ชั้นที่ 2
        )

        # สอง layer นี้แปลง embedding ไปเป็นพารามิเตอร์ FiLM
        self.to_gamma = nn.Linear(32, ch)
        self.to_beta = nn.Linear(32, ch)

        # body เป็น convolution หลักหลังจาก feature ถูกปรับตาม genre แล้ว
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),  # convolution ชั้นที่ 1 ใช้สกัด feature เพิ่มเติมโดยคงจำนวน channel และขนาดเดิม
            nn.GELU(),  # activation function เพิ่มความสามารถในการเรียนรู้ความสัมพันธ์ที่ซับซ้อน
            nn.Conv2d(ch, ch, 3, padding=1),  # convolution ชั้นที่ 2 ประมวลผล feature ต่อจากชั้นก่อนหน้า
            nn.GELU(),  # activation function หลัง convolution ชั้นที่ 2
            nn.Conv2d(ch, ch, 3, padding=1),  # convolution ชั้นที่ 3 เพื่อสกัด feature ให้ลึกขึ้นอีก
            nn.GELU(),  # activation function หลัง convolution ชั้นสุดท้ายใน body
        )

        # head แปลง latent feature กลับเป็น mel 1 channel
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
        # ดึง feature ของสเปกโตรแกรมก่อน
        h = self.stem(x)
        # แปลง genre id เป็น embedding เพื่อใช้เป็นเงื่อนไขของโมเดล
        g = self.emb(gid)

        # สร้างพารามิเตอร์ FiLM สำหรับ scale และ shift feature map
        gamma = self.to_gamma(g).unsqueeze(-1).unsqueeze(-1)  # แปลง embedding เป็นค่า scale แล้วเพิ่มมิติท้ายให้ broadcast กับ feature map ได้
        beta = self.to_beta(g).unsqueeze(-1).unsqueeze(-1)  # แปลง embedding เป็นค่า shift แล้วเพิ่มมิติท้ายให้ broadcast กับ feature map ได้

        # ปรับ feature map ตาม genre แล้วค่อยส่งผ่าน convolution body
        h = h * (1 + gamma) + beta  # ปรับ feature map ด้วยค่า scale และ shift ที่ได้จาก genre conditioning
        h = self.body(h)  # ส่ง feature ที่ถูกปรับแล้วเข้า body เพื่อประมวลผลต่อ
        return x + self.head(h)  # แปลงผลลัพธ์กลับเป็น 1 channel แล้วบวกกับ input เดิมแบบ residual connection

# ใช้แปลง waveform เป็น mel spectrogram หน่วย dB
# มีไว้เตรียมข้อมูลเสียงให้อยู่ในรูปแบบเดียวกับที่โมเดลใช้ตอน train
def waveform_to_mel_db(y: np.ndarray) -> np.ndarray:
    # แปลง waveform เป็น mel spectrogram แบบ dB ให้ตรงกับ representation ที่โมเดลเทรนมา
    # ขั้นตอนนี้คือ preprocessing หลักก่อนส่งข้อมูลเข้าโมเดล
    mel = librosa.feature.melspectrogram(
        y=y,  # สัญญาณเสียงต้นฉบับที่นำมาสร้าง mel spectrogram
        sr=SR,  # sample rate ของเสียง
        n_fft=N_FFT,  # ขนาดหน้าต่าง FFT ที่ใช้แปลงสัญญาณเป็นความถี่
        hop_length=HOP,  # จำนวน sample ที่เลื่อนในแต่ละเฟรม
        n_mels=N_MELS,  # จำนวน mel bins ของผลลัพธ์
        power=2.0,  # ใช้ power spectrogram โดยยกกำลัง 2
    )
    # แปลง power spectrogram เป็น dB เพื่อให้สเกลข้อมูลเสถียรขึ้นสำหรับโมเดล
    mel_db = librosa.power_to_db(mel, ref=1.0)
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)
    return mel_db.astype(np.float32)

# ใช้แปลง mel dB ไปเป็นช่วง 0..1
# มีไว้ป้อนเข้า checkpoint รุ่น FiLM ที่ต้องการ input แบบ normalize แล้ว
def mel_db_to_norm(mel_db: np.ndarray) -> np.ndarray:
    # แปลงช่วง dB ไปอยู่ใน 0-  1 สำหรับ checkpoint แบบ FiLM
    # FiLM checkpoint ถูกฝึกด้วย input ที่ normalize แล้ว จึงต้องแปลงก่อน inference
    mel_db = np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)
    mel_norm = (mel_db + TOP_DB) / TOP_DB
    return np.clip(mel_norm, 0.0, 1.0).astype(np.float32)

# ใช้แปลง mel ที่ถูก normalize แล้วกลับไปเป็นหน่วย dB
# มีไว้แปลง output ของโมเดล FiLM กลับมาใช้ในขั้นตอนสร้างเสียง
def mel_norm_to_db(mel_norm: np.ndarray) -> np.ndarray:
    # แปลงค่าที่อยู่ในช่วง 0..1 กลับไปเป็น dB สำหรับขั้นตอน reconstruct เสียง
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return (mel_norm * TOP_DB - TOP_DB).astype(np.float32)

# ใช้สร้าง waveform กลับจาก mel spectrogram โดยยืม phase จากเสียงต้นฉบับ
# มีไว้ลด artifact จากการ reconstruct เสียงหลังโมเดลทำนาย EQ
def mel_db_to_waveform_with_input_phase(mel_db: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    # สร้าง waveform กลับจาก mel ที่ทำนายได้ โดยยืม phase จากเสียงต้นฉบับ
    # วิธีนี้ใช้ phase ของต้นฉบับช่วย reconstruct เสียงเพื่อลด artifact จากการกู้คลื่นกลับ
    mel = librosa.db_to_power(np.clip(mel_db, MEL_DB_MIN, MEL_DB_MAX)).astype(np.float32)
    mag_pred = librosa.feature.inverse.mel_to_stft(mel, sr=SR, n_fft=N_FFT)
    stft_ref = librosa.stft(y_ref, n_fft=N_FFT, hop_length=HOP)

    if mag_pred.ndim != 2 or stft_ref.ndim != 2:
        raise ValueError("Invalid spectrogram dimensions for phase reconstruction.")

    # ตัดจำนวนเฟรมเวลาให้เท่ากันก่อนประกอบ magnitude ใหม่กับ phase เดิม
    time_bins = min(mag_pred.shape[1], stft_ref.shape[1])
    if time_bins <= 0:
        return np.zeros(len(y_ref), dtype=np.float32)

    mag_pred = mag_pred[:, :time_bins]
    phase = np.angle(stft_ref[:, :time_bins])
    # สร้าง STFT ใหม่จาก magnitude ที่โมเดลพยากรณ์และ phase ของต้นฉบับ
    stft_new = mag_pred * np.exp(1j * phase)
    y = librosa.istft(stft_new, hop_length=HOP, length=len(y_ref))
    return y.astype(np.float32)

# ใช้ปรับความดังเฉลี่ยของเสียงที่ผ่าน Auto-EQ ให้ใกล้กับต้นฉบับ
# มีไว้กันไม่ให้ผลลัพธ์ดังหรือเบาเกินไปเพราะการประมวลผล
def match_loudness_rms(y_ref: np.ndarray, y_out: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # ปรับความดังรวมของเสียงที่ประมวลผลแล้วให้ใกล้กับต้นฉบับ
    ref_rms = np.sqrt(np.mean(y_ref**2)) + eps  # คำนวณค่า RMS ของเสียงต้นฉบับ
    out_rms = np.sqrt(np.mean(y_out**2)) + eps  # คำนวณค่า RMS ของเสียงที่ผ่านการประมวลผล
    gain = ref_rms / out_rms  # หาสัดส่วน gain ที่จะใช้ปรับความดังให้ใกล้เสียงต้นฉบับ
    max_gain = 10.0 ** (MAX_RMS_GAIN_DB / 20.0)  # แปลงเพดาน gain สูงสุดจาก dB เป็นสเกลเชิงเส้น
    gain = float(np.clip(gain, 1.0 / max_gain, max_gain))  # จำกัด gain ไม่ให้เพิ่มหรือลดความดังเกินที่กำหนด
    return y_out * gain  # คืนค่าสัญญาณที่ถูกปรับความดังแล้ว

# ใช้จำกัด peak สูงสุดของสัญญาณเสียง
# มีไว้ลดความเสี่ยง clipping หลังผ่าน Auto-EQ
def limit_peak(y: np.ndarray, peak_limit: float = PEAK_LIMIT) -> np.ndarray:
    # กัน peak ไม่ให้เกินเพดานที่กำหนดเพื่อลดความเสี่ยง clipping
    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0  # หาค่า peak สูงสุดของสัญญาณเสียง
    if peak <= peak_limit or peak <= 0.0:  # ถ้า peak ยังไม่เกินเพดานหรือไม่มีสัญญาณก็ไม่ต้องปรับ
        return y.astype(np.float32)  # คืนค่าสัญญาณเดิมในรูปแบบ float32
    return (y * (peak_limit / peak)).astype(np.float32)  # ลดระดับสัญญาณทั้งก้อนเพื่อให้ peak ไม่เกินที่กำหนด

# ใช้ประเมินว่า chunk นี้ควรปรับ Auto-EQ แรงแค่ไหน
# มีไว้ลด artifact ในช่วงที่เสียงเบามากหรือย่านแหลมจัดเกินไป
def compute_adaptive_strength(chunk: np.ndarray, mel_db: np.ndarray) -> tuple[float, float]:
    # ลดความแรงของการประมวลผลในช่วงที่เงียบมากหรือแหลมมาก เพื่อลด artifact
    # คืนค่า 2 ตัว:
    # - blend_scale ใช้ลดสัดส่วนการผสมเสียงที่ผ่าน Auto-EQ
    # - delta_scale ใช้ลดแรงของ delta EQ ที่โมเดลทำนาย
    chunk_rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size > 0 else 0.0  # คำนวณค่า RMS ของ chunk เสียงนี้
    mel_power = librosa.db_to_power(mel_db)  # แปลง mel จากหน่วย dB กลับเป็นพลังงาน
    if mel_power.ndim != 2 or mel_power.shape[0] < 4:  # ถ้า mel มีมิติไม่ถูกต้องหรือมีจำนวนน้อยเกินไป
        return 1.0, 1.0  # คืนค่า scale ปกติโดยไม่ลดความแรงของการประมวลผล

    # ใช้ช่วงบนประมาณ 35% ของ mel bins เป็นตัวแทนพลังงานย่านแหลม
    split_bin = max(int(mel_power.shape[0] * 0.65), 1)  # หาตำแหน่งเริ่มต้นของย่านความถี่สูงประมาณ 35% ด้านบน
    hf_power = float(np.mean(mel_power[split_bin:, :]))  # คำนวณพลังงานเฉลี่ยของย่านความถี่สูง
    total_power = float(np.mean(mel_power)) + 1e-8  # คำนวณพลังงานเฉลี่ยรวมทั้งหมด
    hf_ratio = hf_power / total_power  # หาสัดส่วนพลังงานย่านสูงเทียบกับพลังงานรวม

    blend_scale = 1.0  # กำหนดค่าเริ่มต้นของสัดส่วนการ blend
    delta_scale = 1.0  # กำหนดค่าเริ่มต้นของสัดส่วนการปรับ delta EQ

    if chunk_rms < LOW_RMS_THRESHOLD:  # ถ้า chunk นี้มีความดังต่ำกว่าค่าที่กำหนด
        quiet_scale = max(MIN_BLEND_SCALE, chunk_rms / max(LOW_RMS_THRESHOLD, 1e-8))  # คำนวณสเกลลดความแรงสำหรับช่วงเสียงเบา
        blend_scale *= quiet_scale  # ลดสัดส่วนการ blend ตามความเบาของเสียง
        delta_scale *= max(MIN_DELTA_SCALE, quiet_scale)  # ลดสัดส่วนการปรับ delta EQ ตามความเบาของเสียง

    if hf_ratio > HIGH_HF_RATIO_THRESHOLD:  # ถ้าย่านความถี่สูงเด่นเกินค่าที่กำหนด
        hf_scale = max(MIN_BLEND_SCALE, HIGH_HF_RATIO_THRESHOLD / max(hf_ratio, 1e-8))  # คำนวณสเกลลดความแรงสำหรับย่านแหลมที่เด่นเกินไป
        blend_scale *= hf_scale  # ลดสัดส่วนการ blend เพื่อลดความคมและ artifact
        delta_scale *= max(MIN_DELTA_SCALE, hf_scale)  # ลดสัดส่วนการปรับ delta EQ สำหรับย่านแหลม

    blend_scale = float(np.clip(blend_scale, MIN_BLEND_SCALE, 1.0))  # จำกัดค่า blend_scale ให้อยู่ในช่วงที่กำหนด
    delta_scale = float(np.clip(delta_scale, MIN_DELTA_SCALE, 1.0))  # จำกัดค่า delta_scale ให้อยู่ในช่วงที่กำหนด
    return blend_scale, delta_scale  # คืนค่าสเกลสำหรับ blend และ delta EQ

# ใช้ smooth ค่าบนแกนที่กำหนดด้วย Hann window
# มีไว้ทำให้เส้น EQ ที่โมเดลทำนายไม่หักหรือแกว่งแรงเกินไป
def _smooth_axis(arr: np.ndarray, axis: int, taps: int) -> np.ndarray:
    # smoothing ด้วย Hann window ตามแกนที่กำหนด เพื่อไม่ให้ EQ เปลี่ยนแบบหักมุมเกินไป
    if taps <= 1:
        return arr

    # บังคับให้ taps เป็นจำนวนคี่ เพื่อให้ kernel มีจุดกึ่งกลางชัดเจน
    taps = max(1, int(taps))  # บังคับให้ taps เป็นจำนวนเต็มและมีค่าอย่างน้อย 1
    if taps % 2 == 0:  # ถ้า taps เป็นเลขคู่
        taps += 1  # เพิ่มให้เป็นเลขคี่เพื่อให้ kernel มีจุดกึ่งกลาง

    kernel = np.hanning(taps).astype(np.float32)  # สร้าง Hann window สำหรับใช้เป็น smoothing kernel
    kernel_sum = float(kernel.sum())  # คำนวณผลรวมของค่าน้ำหนักใน kernel
    if kernel_sum <= 0.0:  # ถ้าผลรวมของ kernel ไม่ถูกต้อง
        return arr  # คืนค่าเดิมโดยไม่ทำ smoothing
    kernel /= kernel_sum  # normalize kernel ให้ผลรวมเท่ากับ 1

    pad = taps // 2  # คำนวณจำนวนช่องที่ต้อง pad ทั้งสองด้าน
    pad_width = [(0, 0)] * arr.ndim  # สร้างค่า pad เริ่มต้นสำหรับทุกมิติ
    pad_width[axis] = (pad, pad)  # กำหนดให้ pad เฉพาะแกนที่ต้องการ smooth
    padded = np.pad(arr, pad_width, mode="edge")  # pad ขอบด้วยค่าเดิมของปลายสัญญาณ

    return np.apply_along_axis(  # ใช้ convolve ตามแกนที่กำหนดแล้วคืนผลลัพธ์เป็น float32
        lambda v: np.convolve(v, kernel, mode="valid"),  # convolve ข้อมูลหนึ่งแนวกับ kernel
        axis=axis,  # ระบุแกนที่ต้องการประมวลผล
        arr=padded,  # ใช้อาเรย์ที่ pad แล้วเป็นอินพุต
    ).astype(np.float32)  # แปลงผลลัพธ์สุดท้ายเป็น float32

# ใช้ smooth delta EQ ทั้งตามแกนความถี่และแกนเวลา
# มีไว้ทำให้การปรับ EQ ฟังนุ่มและต่อเนื่องขึ้น
def smooth_delta_mel(delta_db: np.ndarray) -> np.ndarray:
    # ทำให้ delta EQ ที่โมเดลทำนายมานุ่มขึ้นทั้งตามความถี่และตามเวลา
    delta_db = _smooth_axis(delta_db, axis=0, taps=DELTA_SMOOTH_FREQ_BINS)
    delta_db = _smooth_axis(delta_db, axis=1, taps=DELTA_SMOOTH_TIME_FRAMES)
    return delta_db.astype(np.float32)

def taper_high_freq_delta(delta_db: np.ndarray) -> np.ndarray:  # ฟังก์ชันลดแรงการปรับในย่านความถี่สูงแบบค่อยเป็นค่อยไป
    if delta_db.ndim != 2 or delta_db.shape[0] <= 1:  # ถ้าอินพุตไม่ได้เป็นเมทริกซ์ 2 มิติหรือมีความถี่น้อยเกินไป
        return delta_db.astype(np.float32)  # คืนค่าเดิมในรูปแบบ float32

    taper = np.ones(delta_db.shape[0], dtype=np.float32)  # สร้างเวกเตอร์ค่าน้ำหนักเริ่มต้นเป็น 1 ทุกตำแหน่ง
    start_idx = int(np.clip(HF_TAPER_START, 0.0, 1.0) * (delta_db.shape[0] - 1))  # หาจุดเริ่มต้นของการลดแรงในย่านสูง
    if start_idx < delta_db.shape[0] - 1:  # ถ้ามีช่วงให้ค่อย ๆ ลดแรงต่อจนถึงปลายย่านความถี่
        taper[start_idx:] = np.linspace(1.0, HF_TAPER_END, delta_db.shape[0] - start_idx, dtype=np.float32)  # สร้างค่าน้ำหนักไล่ระดับจาก 1.0 ไปถึงค่าปลายที่กำหนด
    return (delta_db * taper[:, None]).astype(np.float32)  # คูณ delta_db กับค่าน้ำหนักในแต่ละย่านแล้วคืนผลลัพธ์

def build_chunk_window(chunk_len: int, fade_len: int) -> np.ndarray:  # ฟังก์ชันสร้างหน้าต่างน้ำหนักสำหรับ crossfade ระหว่าง chunk
    if chunk_len <= 1:  # ถ้าความยาว chunk น้อยเกินไป
        return np.ones((max(chunk_len, 1),), dtype=np.float32)  # คืนหน้าต่างที่มีค่า 1 ทั้งหมด

    fade_len = min(max(int(fade_len), 0), chunk_len // 2)  # จำกัดความยาวช่วง fade ให้อยู่ในช่วงที่ปลอดภัย
    if fade_len <= 0:  # ถ้าไม่มีช่วง fade
        return np.ones((chunk_len,), dtype=np.float32)  # คืนหน้าต่างที่มีค่า 1 ทั้งหมด

    window = np.ones((chunk_len,), dtype=np.float32)  # สร้างหน้าต่างเริ่มต้นที่มีค่าน้ำหนักเท่ากับ 1
    fade = np.hanning(fade_len * 2).astype(np.float32)  # สร้าง Hann window สำหรับใช้ทำ fade-in และ fade-out
    fade_in = fade[:fade_len]  # ตัดครึ่งหน้าไว้ใช้เป็นช่วง fade-in
    fade_out = fade[fade_len:]  # ตัดครึ่งหลังไว้ใช้เป็นช่วง fade-out
    window[:fade_len] = np.maximum(window[:fade_len] * fade_in, 1e-4)  # กำหนดค่าน้ำหนักช่วงต้นให้ค่อย ๆ เพิ่มขึ้น
    window[-fade_len:] = np.maximum(window[-fade_len:] * fade_out, 1e-4)  # กำหนดค่าน้ำหนักช่วงท้ายให้ค่อย ๆ ลดลง
    return window  # คืนหน้าต่างน้ำหนักที่พร้อมใช้กับการ overlap-add

def extract_model_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:  # ฟังก์ชันดึง state_dict ออกจาก checkpoint ไม่ว่าจะเก็บแบบใด
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:  # กรณี checkpoint ห่อ state_dict ไว้ในคีย์ state_dict
        state = checkpoint["state_dict"]  # ดึง state_dict ที่อยู่ภายในออกมาใช้งาน
    elif isinstance(checkpoint, Mapping):  # กรณีไฟล์บันทึก state_dict ตรง ๆ
        state = checkpoint  # ใช้ checkpoint ทั้งก้อนเป็น state_dict
    else:  # กรณีรูปแบบ checkpoint ไม่ถูกต้อง
        raise AutoEQModelLoadError("Checkpoint format is invalid (expected mapping or mapping['state_dict']).")  # แจ้งข้อผิดพลาดเรื่องรูปแบบ checkpoint

    if not isinstance(state, Mapping):  # ตรวจสอบว่า state ที่ได้ออกมายังเป็น mapping อยู่หรือไม่
        raise AutoEQModelLoadError("Checkpoint 'state_dict' is not a mapping.")  # แจ้งข้อผิดพลาดถ้า state_dict ไม่ใช่ mapping
    return state  # คืนค่า state_dict ที่พร้อมนำไปโหลดเข้าโมเดล

def normalize_state_dict_keys(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # ฟังก์ชันปรับชื่อ key ของ state_dict ให้ตรงกับโมเดลปัจจุบัน
    normalized: dict[str, torch.Tensor] = {}  # สร้าง dict ใหม่ไว้เก็บ key ที่ถูกปรับชื่อแล้ว
    for key, value in state.items():  # วนดูพารามิเตอร์ทุกตัวใน state_dict
        if not isinstance(key, str) or not torch.is_tensor(value):  # ข้ามรายการที่ key ไม่ใช่สตริงหรือ value ไม่ใช่ tensor
            continue  # ไม่เอารายการนี้ไปใช้ต่อ

        new_key = key  # เริ่มต้นด้วยชื่อ key เดิมก่อน
        if new_key.startswith("module."):  # ถ้า key มี prefix จาก DataParallel
            new_key = new_key[len("module.") :]  # ตัด prefix module. ออก
        if new_key.startswith("_orig_mod."):  # ถ้า key มี prefix จาก torch.compile
            new_key = new_key[len("_orig_mod.") :]  # ตัด prefix _orig_mod. ออก
        if new_key.startswith("net."):  # ถ้า key ใช้ชื่อแบบโมเดลเก่า
            new_key = "body." + new_key[len("net.") :]  # map จาก net. ไปเป็น body. ให้ตรงกับโค้ดปัจจุบัน

        normalized[new_key] = value  # เก็บ tensor ลงใน dict ใหม่ด้วยชื่อ key ที่ปรับแล้ว

    if not normalized:  # ถ้าไม่มี tensor ไหนถูกเก็บไว้เลย
        raise AutoEQModelLoadError("No tensor parameters found in checkpoint state dict.")  # แจ้งว่าไม่พบพารามิเตอร์ที่ใช้ได้ใน checkpoint
    return normalized  # คืน state_dict ที่ปรับชื่อ key แล้ว

def infer_legacy_arch_from_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:  # ฟังก์ชันเดา architecture ของโมเดล legacy จาก shape ของ weight
    required = ("body.0.weight", "body.3.weight", "body.6.weight", "body.9.weight")  # รายชื่อ weight ที่ต้องมีเพื่อใช้เดาโครงสร้างโมเดลเก่า
    missing = [key for key in required if key not in state]  # ตรวจว่ามี key ไหนหายไปจาก checkpoint บ้าง
    if missing:  # ถ้าขาด weight ที่จำเป็น
        raise AutoEQModelLoadError(f"Checkpoint missing required layer weights: {', '.join(missing)}")  # แจ้งว่า checkpoint ขาด layer ที่ต้องใช้

    w0 = state["body.0.weight"]  # ดึงน้ำหนัก convolution ชั้นแรกของ body
    w3 = state["body.3.weight"]  # ดึงน้ำหนัก convolution ชั้นถัดไปของ body
    w6 = state["body.6.weight"]  # ดึงน้ำหนัก convolution ชั้นถัดไปของ body
    w9 = state["body.9.weight"]  # ดึงน้ำหนัก convolution ชั้นสุดท้ายของ body

    if any(weight.ndim != 4 for weight in (w0, w3, w6, w9)):  # ตรวจว่าน้ำหนัก convolution ทุกตัวเป็น tensor 4 มิติหรือไม่
        raise AutoEQModelLoadError("Checkpoint contains invalid convolution weight dimensions.")  # แจ้งข้อผิดพลาดถ้ามิติน้ำหนักไม่ถูกต้อง

    ch1 = int(w0.shape[0])  # ใช้จำนวน output channels ของชั้นแรกเป็นค่า channel ชุดที่ 1
    ch2 = int(w3.shape[0])  # ใช้จำนวน output channels ของชั้นที่สองเป็นค่า channel ชุดที่ 2
    ch3 = int(w6.shape[0])  # ใช้จำนวน output channels ของชั้นที่สามเป็นค่า channel ชุดที่ 3

    if int(w3.shape[1]) != ch1:  # ตรวจว่า input channels ของชั้นที่สองตรงกับ output ของชั้นแรกหรือไม่
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.3.weight input channels do not match body.0.")  # แจ้งข้อผิดพลาดถ้า shape ต่อกันไม่ได้
    if int(w6.shape[1]) != ch2:  # ตรวจว่า input channels ของชั้นที่สามตรงกับ output ของชั้นก่อนหน้าหรือไม่
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.6.weight input channels do not match body.3.")  # แจ้งข้อผิดพลาดถ้า shape ต่อกันไม่ได้
    if int(w9.shape[1]) != ch3 or int(w9.shape[0]) != 1:  # ตรวจว่าชั้นสุดท้ายรับ channel ถูกต้องและออก 1 channel หรือไม่
        raise AutoEQModelLoadError("Checkpoint shape mismatch: body.9.weight is incompatible with inferred channels.")  # แจ้งข้อผิดพลาดถ้า shape ไม่เข้ากับ architecture ที่เดาได้

    return ch1, ch2, ch3  # คืนค่าจำนวน channel ของโมเดล legacy ที่เดาได้


def infer_film_arch_from_state_dict(state: Mapping[str, torch.Tensor]) -> tuple[int, int]:  # ฟังก์ชันเดา architecture ของโมเดล FiLM จาก shape ของ weight
    required = (  # รายชื่อ weight ที่ต้องมีเพื่อใช้เดาโครงสร้างโมเดล FiLM
        "emb.weight",
        "stem.0.weight",
        "stem.2.weight",
        "to_gamma.weight",
        "to_beta.weight",
        "body.0.weight",
        "body.2.weight",
        "body.4.weight",
        "head.weight",
    )
    missing = [key for key in required if key not in state]  # ตรวจว่ามี key สำคัญตัวไหนหายไปบ้าง
    if missing:  # ถ้าขาด weight ที่จำเป็น
        raise AutoEQModelLoadError(f"Checkpoint missing required FiLM layer weights: {', '.join(missing)}")  # แจ้งว่า checkpoint ขาด layer ของ FiLM

    emb_weight = state["emb.weight"]  # ดึงน้ำหนัก embedding ของ genre
    stem0 = state["stem.0.weight"]  # ดึงน้ำหนัก convolution ชั้นแรกของ stem
    stem2 = state["stem.2.weight"]  # ดึงน้ำหนัก convolution ชั้นที่สองของ stem
    gamma = state["to_gamma.weight"]  # ดึงน้ำหนัก linear ที่ใช้สร้าง gamma
    beta = state["to_beta.weight"]  # ดึงน้ำหนัก linear ที่ใช้สร้าง beta
    body0 = state["body.0.weight"]  # ดึงน้ำหนัก convolution ชั้นแรกของ body
    body2 = state["body.2.weight"]  # ดึงน้ำหนัก convolution ชั้นที่สองของ body
    body4 = state["body.4.weight"]  # ดึงน้ำหนัก convolution ชั้นที่สามของ body
    head = state["head.weight"]  # ดึงน้ำหนัก convolution ของ head

    if any(weight.ndim != 2 for weight in (emb_weight, gamma, beta)):  # ตรวจว่า embedding และ linear weights เป็น tensor 2 มิติหรือไม่
        raise AutoEQModelLoadError("Checkpoint contains invalid FiLM linear/embedding weight dimensions.")  # แจ้งข้อผิดพลาดถ้ามิติน้ำหนัก linear ไม่ถูกต้อง
    if any(weight.ndim != 4 for weight in (stem0, stem2, body0, body2, body4, head)):  # ตรวจว่า convolution weights เป็น tensor 4 มิติหรือไม่
        raise AutoEQModelLoadError("Checkpoint contains invalid FiLM convolution weight dimensions.")  # แจ้งข้อผิดพลาดถ้ามิติน้ำหนัก convolution ไม่ถูกต้อง

    n_genres = int(emb_weight.shape[0])  # จำนวน genre ที่ checkpoint รองรับ
    emb_dim = int(emb_weight.shape[1])  # ขนาด embedding vector ของแต่ละ genre
    ch = int(stem0.shape[0])  # จำนวน latent channels ของโมเดล

    if emb_dim != 32:  # ตรวจว่าขนาด embedding ตรงกับที่โค้ดรองรับหรือไม่
        raise AutoEQModelLoadError(f"Unsupported FiLM embedding size: expected 32, got {emb_dim}")  # แจ้งข้อผิดพลาดถ้าขนาด embedding ไม่ตรง
    if int(stem0.shape[1]) != 1 or int(stem2.shape[0]) != ch or int(stem2.shape[1]) != ch:  # ตรวจว่า shape ของ stem ต่อกันได้ถูกต้องหรือไม่
        raise AutoEQModelLoadError("Checkpoint stem shape mismatch.")  # แจ้งข้อผิดพลาดถ้า stem shape ไม่ตรง
    if int(gamma.shape[0]) != ch or int(gamma.shape[1]) != emb_dim:  # ตรวจว่า shape ของ to_gamma ตรงกับ channel และ embedding หรือไม่
        raise AutoEQModelLoadError("Checkpoint to_gamma shape mismatch.")  # แจ้งข้อผิดพลาดถ้า to_gamma shape ไม่ตรง
    if int(beta.shape[0]) != ch or int(beta.shape[1]) != emb_dim:  # ตรวจว่า shape ของ to_beta ตรงกับ channel และ embedding หรือไม่
        raise AutoEQModelLoadError("Checkpoint to_beta shape mismatch.")  # แจ้งข้อผิดพลาดถ้า to_beta shape ไม่ตรง
    if any(int(weight.shape[0]) != ch or int(weight.shape[1]) != ch for weight in (body0, body2, body4)):  # ตรวจว่า convolution ใน body ใช้ channel ขนาดเดียวกันทุกชั้นหรือไม่
        raise AutoEQModelLoadError("Checkpoint body convolution shape mismatch.")  # แจ้งข้อผิดพลาดถ้า body shape ไม่ตรง
    if int(head.shape[0]) != 1 or int(head.shape[1]) != ch:  # ตรวจว่า head ออก 1 channel และรับ channel ถูกต้องหรือไม่
        raise AutoEQModelLoadError("Checkpoint head shape mismatch.")  # แจ้งข้อผิดพลาดถ้า head shape ไม่ตรง

    return n_genres, ch  # คืนจำนวน genre และจำนวน channel ของโมเดล FiLM ที่เดาได้


@lru_cache(maxsize=1)  # cache ผลลัพธ์การโหลดโมเดลไว้ 1 ชุดเพื่อลดการโหลดซ้ำ
def load_auto_eq_model(device: str = "cpu") -> nn.Module:  # ฟังก์ชันโหลดโมเดล Auto-EQ จาก checkpoint
    try:  # ครอบการโหลดโมเดลไว้เพื่อจัดการ error ให้ชัดเจน
        checkpoint = torch.load(MODEL_PATH, map_location=device)  # โหลด checkpoint จากไฟล์มายังอุปกรณ์ที่กำหนด
        raw_state = extract_model_state_dict(checkpoint)  # ดึง state_dict ออกจาก checkpoint
        state = normalize_state_dict_keys(raw_state)  # ปรับชื่อ key ให้ตรงกับโมเดลปัจจุบัน

        if "emb.weight" in state:  # ถ้ามี embedding แปลว่าเป็น checkpoint แบบ FiLM
            n_genres, ch = infer_film_arch_from_state_dict(state)  # เดาจำนวน genre และ channel จาก checkpoint
            logger.info(  # บันทึก log ข้อมูลการโหลดโมเดล
                "Loading Auto-EQ FiLM model path=%s device=%s genres=%d channels=%d",
                MODEL_PATH,
                device,
                n_genres,
                ch,
            )
            model = AutoEQ(n_genres=n_genres, ch=ch)  # สร้างโมเดล AutoEQ ให้ตรงกับ checkpoint
            model.requires_gid = True  # ระบุว่าโมเดลนี้ต้องใช้ genre id ตอน inference
            model.input_representation = "mel_norm"  # ระบุว่าโมเดลนี้รับ input แบบ mel ที่ normalize แล้ว

        model.load_state_dict(state, strict=True)  # โหลดน้ำหนักจาก state_dict เข้าโมเดลแบบ strict
        model.to(device)  # ย้ายโมเดลไปยังอุปกรณ์ที่กำหนด
        model.eval()  # สลับโมเดลเป็นโหมด inference
        return model  # คืนโมเดลที่พร้อมใช้งาน
    except AutoEQModelLoadError:  # ถ้าเป็น error ที่เรารู้จักอยู่แล้ว
        raise  # ส่ง error เดิมต่อไป
    except Exception as exc:  # ถ้าเป็น error อื่นระหว่างโหลดโมเดล
        raise AutoEQModelLoadError(f"Failed to load Auto-EQ model from {MODEL_PATH}: {exc}") from exc  # ห่อเป็น error ของระบบนี้เพื่อให้อ่านง่ายขึ้น


def resolve_genre_id(model: nn.Module, genre: str | int | None) -> int | None:  # ฟังก์ชันแปลง genre ที่รับเข้ามาให้เป็น genre id
    if not getattr(model, "requires_gid", False):  # ถ้าโมเดลไม่ต้องใช้ genre id
        return None  # คืน None ได้เลย

    if genre is None:  # ถ้า caller ไม่ส่ง genre มา
        genre = DEFAULT_GENRE  # ใช้ genre เริ่มต้นแทน
        logger.warning("No genre provided for FiLM checkpoint. Falling back to default genre=%s", genre)  # แจ้งเตือนว่า fallback ไปใช้ genre เริ่มต้น

    if isinstance(genre, int):  # ถ้า genre ถูกส่งมาเป็นตัวเลขอยู่แล้ว
        if 0 <= genre < len(GENRES):  # ตรวจว่า genre id อยู่ในช่วงที่รองรับหรือไม่
            return genre  # คืนค่า genre id นั้นได้เลย
        raise ValueError(f"Genre id out of range: {genre}. Valid range is 0..{len(GENRES) - 1}.")  # แจ้งข้อผิดพลาดถ้าเลข genre เกินช่วง

    genre_key = str(genre).strip().lower()  # แปลง genre ให้เป็นสตริงมาตรฐานสำหรับ lookup
    if genre_key not in GENRE2ID:  # ถ้าไม่พบชื่อ genre ในรายการที่รองรับ
        raise ValueError(f"Unknown genre '{genre}'. Valid genres: {', '.join(GENRES)}")  # แจ้งข้อผิดพลาดพร้อมรายชื่อ genre ที่ใช้ได้
    return GENRE2ID[genre_key]  # คืน genre id ที่ตรงกับชื่อ genre


def predict_mel_db(  # ฟังก์ชันเรียกโมเดลเพื่อทำนาย mel spectrogram หลัง Auto-EQ
    model: nn.Module,  # โมเดลที่โหลดไว้และพร้อมทำ inference
    mel_db: np.ndarray,  # mel spectrogram อินพุตหน่วย dB
    device: str,  # อุปกรณ์ที่ใช้รัน inference
    genre_id: int | None,  # genre id สำหรับโมเดล FiLM หรือ None สำหรับโมเดลที่ไม่ใช้
) -> np.ndarray:  # คืนผลทำนายเป็น mel spectrogram หน่วย dB
    if getattr(model, "input_representation", "mel_db") == "mel_norm":  # ถ้าโมเดลต้องการ input แบบ normalized mel
        model_input = mel_db_to_norm(mel_db)  # แปลง mel จาก dB ไปเป็นช่วง 0..1
        mel_tensor = torch.from_numpy(model_input).unsqueeze(0).unsqueeze(0).to(device)  # แปลงเป็น tensor และเพิ่ม batch/channel dimension
        gid_tensor = torch.tensor([genre_id], dtype=torch.long, device=device)  # สร้าง tensor ของ genre id สำหรับส่งเข้าโมเดล
        with torch.no_grad():  # ปิดการคำนวณ gradient ระหว่าง inference
            pred_mel_norm = model(mel_tensor, gid_tensor).squeeze(0).squeeze(0).cpu().numpy()  # รันโมเดลแล้วดึงผลลัพธ์กลับเป็น numpy
        return mel_norm_to_db(pred_mel_norm)  # แปลงผลลัพธ์จาก normalized mel กลับเป็น dB

    mel_tensor = torch.from_numpy(mel_db).unsqueeze(0).unsqueeze(0).to(device)  # แปลง mel_db เป็น tensor และเพิ่ม batch/channel dimension
    with torch.no_grad():  # ปิดการคำนวณ gradient ระหว่าง inference
        pred_mel_db = model(mel_tensor).squeeze(0).squeeze(0).cpu().numpy()  # รันโมเดล legacy แล้วดึงผลลัพธ์กลับเป็น numpy
    return pred_mel_db.astype(np.float32)  # คืนผลลัพธ์เป็น float32


def load_audio_preserve_channels(input_path: str) -> np.ndarray:  # ฟังก์ชันอ่านไฟล์เสียงโดยคงจำนวน channel เดิมไว้
    audio, sr = sf.read(input_path, dtype="float32", always_2d=True)  # อ่านไฟล์เสียงเป็น float32 และบังคับให้มีมิติ channel เสมอ
    audio = audio.T  # สลับแกนจาก (samples, channels) เป็น (channels, samples)
    if sr != SR:  # ถ้า sample rate ของไฟล์ไม่ตรงกับที่โมเดลต้องการ
        audio = np.stack(  # resample ทีละ channel แล้วรวมกลับเป็นอาเรย์เดียว
            [librosa.resample(ch, orig_sr=sr, target_sr=SR).astype(np.float32) for ch in audio],  # resample แต่ละ channel ไปเป็น sample rate มาตรฐาน
            axis=0,  # วาง channel ไว้บนแกนแรกเหมือนเดิม
        )
    return audio.astype(np.float32)  # คืนข้อมูลเสียงในรูปแบบ float32


def apply_auto_eq_waveform(  # ฟังก์ชันทำ Auto-EQ กับ waveform แบบ mono ทีละช่วง
    y: np.ndarray,  # สัญญาณเสียง mono อินพุต
    model: nn.Module,  # โมเดล Auto-EQ ที่ใช้ทำ inference
    device: str,  # อุปกรณ์ที่ใช้รันโมเดล
    genre_id: int | None,  # genre id สำหรับโมเดล FiLM หรือ None
) -> np.ndarray:  # คืน waveform ที่ผ่านการทำ Auto-EQ แล้ว
    y = np.asarray(y, dtype=np.float32)  # แปลงอินพุตเป็น numpy array ชนิด float32
    if y.ndim != 1:  # ตรวจว่าอินพุตเป็น mono waveform หรือไม่
        raise ValueError("apply_auto_eq_waveform expects a mono waveform.")  # แจ้งข้อผิดพลาดถ้าอินพุตไม่ใช่ mono

    segment_samples = int(SEGMENT_SECONDS * SR)  # คำนวณจำนวน sample ต่อ chunk
    overlap_samples = min(int(OVERLAP_SECONDS * SR), max(segment_samples // 2, 1))  # คำนวณจำนวน sample ที่ใช้ overlap ระหว่าง chunk
    step_samples = max(segment_samples - overlap_samples, 1)  # คำนวณระยะเลื่อนในแต่ละรอบของการประมวลผล

    mixed = np.zeros(len(y), dtype=np.float32)  # เตรียมอาเรย์สำหรับสะสมผลลัพธ์ของทุก chunk
    weight = np.zeros(len(y), dtype=np.float32)  # เตรียมอาเรย์สำหรับสะสมน้ำหนักของแต่ละตำแหน่ง

    for start in range(0, len(y), step_samples):  # วนประมวลผลทีละ chunk ตามระยะเลื่อนที่กำหนด
        end = min(start + segment_samples, len(y))  # หาตำแหน่งสิ้นสุดของ chunk ปัจจุบัน
        chunk = y[start:end]  # ตัดสัญญาณช่วงปัจจุบันออกมาเป็น chunk
        if chunk.size == 0:  # ถ้า chunk ว่าง
            continue  # ข้ามไป chunk ถัดไป

        mel_db = waveform_to_mel_db(chunk)  # แปลง chunk เป็น mel spectrogram หน่วย dB
        pred_mel_db = predict_mel_db(model, mel_db, device, genre_id)  # ให้โมเดลทำนาย mel spectrogram ใหม่
        blend_scale, delta_scale = compute_adaptive_strength(chunk, mel_db)  # คำนวณสเกลปรับความแรงแบบ adaptive

        delta = pred_mel_db - mel_db  # คำนวณความต่างระหว่าง mel ที่ทำนายกับ mel เดิม
        delta = smooth_delta_mel(delta)  # ทำให้เส้น delta เรียบขึ้น
        delta = taper_high_freq_delta(delta)  # ลดแรงการปรับในย่านความถี่สูง
        delta *= delta_scale  # คูณสเกลเพื่อลดหรือเพิ่มความแรงของ delta
        delta = np.clip(delta, -DELTA_CLAMP_DB, DELTA_CLAMP_DB)  # จำกัด delta ไม่ให้เกินขอบเขตที่กำหนด
        mel_out = np.clip(mel_db + delta, MEL_DB_MIN, MEL_DB_MAX)  # รวม delta เข้ากับ mel เดิมและ clip ช่วง dB ให้อยู่ในขอบเขต

        enhanced_chunk = mel_db_to_waveform_with_input_phase(mel_out, chunk)  # สร้าง waveform กลับจาก mel ที่ปรับแล้วโดยอ้างอิง phase เดิม
        if enhanced_chunk.shape[0] != chunk.shape[0]:  # ถ้าความยาวผลลัพธ์ไม่เท่ากับ chunk เดิม
            if enhanced_chunk.shape[0] > chunk.shape[0]:  # ถ้าผลลัพธ์ยาวเกินไป
                enhanced_chunk = enhanced_chunk[: chunk.shape[0]]  # ตัดให้ยาวเท่ากับ chunk เดิม
            else:  # ถ้าผลลัพธ์สั้นเกินไป
                pad = chunk.shape[0] - enhanced_chunk.shape[0]  # คำนวณจำนวน sample ที่ต้องเติม
                enhanced_chunk = np.pad(enhanced_chunk, (0, pad))  # เติมศูนย์ด้านท้ายให้ยาวเท่ากับ chunk เดิม

        output_blend = OUTPUT_BLEND * blend_scale  # คำนวณสัดส่วนการ blend สุดท้ายสำหรับ chunk นี้
        enhanced_chunk = ((1.0 - output_blend) * chunk + output_blend * enhanced_chunk).astype(np.float32)  # ผสมเสียงต้นฉบับกับเสียงที่ผ่าน Auto-EQ
        enhanced_chunk = match_loudness_rms(chunk, enhanced_chunk)  # ปรับความดังรวมให้ใกล้กับต้นฉบับ
        enhanced_chunk = limit_peak(enhanced_chunk)  # จำกัด peak ไม่ให้สูงเกินไป
        enhanced_chunk = np.clip(enhanced_chunk, -1.0, 1.0).astype(np.float32)  # clip ค่าสัญญาณให้อยู่ในช่วงเสียงมาตรฐาน

        fade = min(overlap_samples, chunk.shape[0] // 2)  # กำหนดความยาวช่วง fade สำหรับ chunk นี้
        chunk_weight = build_chunk_window(chunk.shape[0], fade)  # สร้างหน้าต่างน้ำหนักสำหรับ overlap-add
        if start == 0:  # ถ้าเป็น chunk แรกของไฟล์
            chunk_weight[:fade] = 1.0  # ไม่ต้อง fade-in ที่ช่วงต้นสุดของไฟล์
        if end >= len(y):  # ถ้าเป็น chunk สุดท้ายของไฟล์
            chunk_weight[-fade:] = 1.0  # ไม่ต้อง fade-out ที่ช่วงท้ายสุดของไฟล์

        mixed[start:end] += enhanced_chunk * chunk_weight  # สะสมผลลัพธ์ของ chunk นี้ลงในบัฟเฟอร์รวม
        weight[start:end] += chunk_weight  # สะสมน้ำหนักของ chunk นี้ไว้สำหรับเฉลี่ยตอนท้าย

    return np.divide(mixed, np.maximum(weight, 1e-8), out=np.zeros_like(mixed), where=weight > 0)  # เฉลี่ยผลลัพธ์ตามน้ำหนักในช่วง overlap แล้วคืน waveform สุดท้าย


def apply_auto_eq_file(input_path: str, output_path: str, genre: str | int | None = None) -> str:  # ฟังก์ชันทำ Auto-EQ ระดับไฟล์ครบทั้งอ่าน ประมวลผล และเขียนผลลัพธ์
    audio = load_audio_preserve_channels(input_path)  # อ่านไฟล์เสียงพร้อมคงจำนวน channel เดิมไว้

    output_dir = os.path.dirname(output_path)  # หา path ของโฟลเดอร์ปลายทาง
    if output_dir:  # ถ้ามีการระบุโฟลเดอร์ปลายทางไว้
        os.makedirs(output_dir, exist_ok=True)  # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มีอยู่

    device = "cuda" if torch.cuda.is_available() else "cpu"  # เลือกใช้ GPU ถ้ามี ไม่งั้นใช้ CPU
    model = load_auto_eq_model(device)  # โหลดโมเดล Auto-EQ ไปยังอุปกรณ์ที่เลือก
    genre_id = resolve_genre_id(model, genre)  # แปลง genre ที่รับเข้ามาให้เป็น genre id

    enhanced_channels = [  # ประมวลผล Auto-EQ ทีละ channel แล้วเก็บผลลัพธ์ไว้
        apply_auto_eq_waveform(channel, model, device, genre_id)
        for channel in audio
    ]
    enhanced = np.stack(enhanced_channels, axis=0)  # รวม channel ที่ประมวลผลแล้วกลับเป็นอาเรย์เดียว
    sf.write(output_path, enhanced.T, SR)  # เขียนผลลัพธ์ลงไฟล์ด้วย sample rate มาตรฐานของโมเดล
    return output_path  # คืน path ของไฟล์ผลลัพธ์
